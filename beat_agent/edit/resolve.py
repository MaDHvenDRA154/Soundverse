"""Resolve musical addresses against the current grid -> output sample ranges.

All failures raise EditError with a user-facing message that states the real
limits ("this track has bars 1-92"). Nothing is ever clamped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from beat_agent.analysis.schema import BeatGrid
from beat_agent.edit import ops as O


class EditError(ValueError):
    pass


@dataclass
class Span:
    a: int                    # output samples, half-open
    b: int
    what: str                 # how the user addressed it, e.g. "bars 5-8"
    notes: list = field(default_factory=list)


def _beat_limits(g: BeatGrid) -> str:
    if not g.beats:
        return "This track has no beat grid, so beats and bars cannot be addressed."
    return f"This track has beats {g.beats[0].index}-{g.beats[-1].index} ({len(g.beats)} beats)."


def _bar_limits(g: BeatGrid) -> str:
    bars = [b.bar for b in g.bars]
    if not bars:
        return "This track has no bars (the meter could not be determined)."
    lo, hi = min(b for b in bars if b >= 1) if any(b >= 1 for b in bars) else 0, max(bars)
    pick = " plus a pickup bar 0" if 0 in bars else ""
    return f"This track has bars {lo}-{hi}{pick} ({hi - lo + 1} bars)."


def _bar(g: BeatGrid, n: int):
    b = g.bar(n)
    if b is None:
        raise EditError(f"Bar {n} does not exist. {_bar_limits(g)}")
    return b


def _beat_sample(g: BeatGrid, i: int, sr: int, total: int) -> int:
    b = g.beat(i)
    if b is not None:
        return int(round(b.time * sr))
    if g.beats and i == g.beats[-1].index + 1:   # the end of the last beat
        t = g.beats[-1].time + (g.beats[-1].time - g.beats[-2].time if len(g.beats) > 1 else 0.0)
        return min(total, int(round(t * sr)))
    raise EditError(f"Beat {i} does not exist. {_beat_limits(g)}")


def _need_bars(g: BeatGrid):
    if not g.bars:
        raise EditError("Bars are not defined for this track (the meter could not be determined); "
                        "address it in beats instead, or tell me the meter (e.g. 'it's in 4/4').")


def resolve_range(r, g: BeatGrid, sr: int, total: int) -> Span:
    if isinstance(r, O.WholeTrack):
        return Span(0, total, "the whole track")
    if isinstance(r, O.BeatRange):
        if not g.beats:
            raise EditError(_beat_limits(g))
        if r.end <= r.start:
            raise EditError(f"'beat {r.start} to beat {r.end}' is empty: ranges run from the first beat up to "
                            f"(not including) the second, so the second must be larger.")
        a = _beat_sample(g, r.start, sr, total)
        b = _beat_sample(g, r.end, sr, total)
        sp = Span(a, b, f"beats {r.start}-{r.end} (beat {r.start} up to, not including, beat {r.end}: {r.end - r.start} beats)")
        first = g.beats[0]
        if r.start == first.index and first.time > 0.05:
            sp.notes.append(f"Audio before beat {r.start} (0.000-{first.time:.3f} s, no beats there) is not part of the range.")
        elif r.start == 0 and first.index < 0:
            sp.notes.append(f"Beats {first.index} to -1 are a pickup before bar 1 (0.000-{a / sr:.3f} s); "
                            "they come before beat 0 and are kept.")
        return sp
    if isinstance(r, O.BarRange):
        _need_bars(g)
        if r.end < r.start:
            raise EditError(f"'bars {r.start}-{r.end}' runs backwards.")
        b0, b1 = _bar(g, r.start), _bar(g, r.end)
        sp = Span(int(round(b0.start * sr)), int(round(b1.end * sr)) if r.end != g.bars[-1].bar else
                  min(total, int(round(b1.end * sr))),
                  f"bar {r.start}" if r.start == r.end else f"bars {r.start}-{r.end} (inclusive, {r.end - r.start + 1} bars)")
        part = [b.bar for b in g.bars if r.start <= b.bar <= r.end and not b.complete]
        if part:
            sp.notes.append(f"Bar(s) {', '.join(map(str, part))} are incomplete (fewer beats than the meter), "
                            "typically because of an earlier non-bar-aligned edit or a pickup.")
        return sp
    if isinstance(r, O.LastBeats):
        if not g.beats or r.count > len(g.beats):
            raise EditError(f"Cannot take the last {r.count} beats. {_beat_limits(g)}")
        first = g.beats[-r.count]
        sp = Span(int(round(first.time * sr)), total,
                  f"the last {r.count} beats (beat {first.index} to the end of the audio)")
        return sp
    if isinstance(r, O.LastBars):
        _need_bars(g)
        bars = [b for b in g.bars if b.bar >= 1]
        if r.count > len(bars):
            raise EditError(f"Cannot take the last {r.count} bars. {_bar_limits(g)}")
        first = bars[-r.count]
        return Span(int(round(first.start * sr)), total, f"the last {r.count} bars (bar {first.bar} to the end of the audio)")
    if isinstance(r, O.BarsBeforeTempoChange):
        _need_bars(g)
        ch = [c for c in g.tempo.changes if c.get("kind") in ("step", "across_gap")]
        if not ch:
            bpm = f" (one tempo, {g.tempo.bpm} BPM, {g.tempo.type})" if g.tempo.bpm else ""
            raise EditError(f"No tempo change was detected in this track{bpm}, so 'the bar before the tempo "
                            "change' does not point anywhere. Name the bars instead.")
        if r.change is None and len(ch) > 1:
            lst = "; ".join(f"#{i + 1}: bar {c['bar']} ({c['time']:.2f} s, {c['from_bpm']}->{c['to_bpm']} BPM)"
                            for i, c in enumerate(ch))
            raise EditError(f"This track has {len(ch)} tempo changes ({lst}). Which one do you mean?")
        k = (r.change or 1) - 1
        if k >= len(ch):
            raise EditError(f"There are only {len(ch)} tempo change(s).")
        c = ch[k]
        tc = c["time"]
        before = [b for b in g.bars if b.bar >= 1 and b.end <= tc + 1e-3]
        if len(before) < r.count:
            raise EditError(f"Only {len(before)} complete bar(s) end before the tempo change at {tc:.2f} s.")
        sel = before[-r.count:]
        sp = Span(int(round(sel[0].start * sr)), int(round(sel[-1].end * sr)),
                  f"{'bar' if r.count == 1 else f'{r.count} bars'} {sel[0].bar}{'' if r.count == 1 else f'-{sel[-1].bar}'}, "
                  f"the last before the tempo change at beat {c['beat_index']} (bar {c['bar']}, {tc:.3f} s, "
                  f"{c['from_bpm']} -> {c['to_bpm']} BPM)")
        if abs(sel[-1].end - tc) > 1e-3:
            sp.notes.append(f"The change falls inside bar {c['bar']}; I took the last bar that ends before it.")
        return sp
    raise EditError(f"Unsupported range {r!r}")


def resolve_point(p, g: BeatGrid, sr: int, total: int) -> tuple[int, str]:
    if isinstance(p, O.AtStart):
        return 0, "the start"
    if isinstance(p, O.AtEnd):
        return total, "the end"
    if isinstance(p, O.AtBeat):
        return _beat_sample(g, p.beat, sr, total), f"before beat {p.beat}"
    if isinstance(p, O.AtBar):
        _need_bars(g)
        return int(round(_bar(g, p.bar).start * sr)), f"before bar {p.bar}"
    if isinstance(p, O.AfterBar):
        _need_bars(g)
        return min(total, int(round(_bar(g, p.bar).end * sr))), f"after bar {p.bar}"
    raise EditError(f"Unsupported point {p!r}")


def describe(g: BeatGrid, a: int, b: int, sr: int) -> str:
    """'beats 16-31 / bars 5-8, 7.742-15.484 s (7.742 s)' for an output span."""
    t0, t1 = a / sr, b / sr
    inside = [x for x in g.beats if t0 - 1e-6 <= x.time < t1 - 1e-6]
    parts = []
    if inside:
        parts.append(f"beats {inside[0].index}-{inside[-1].index}" if len(inside) > 1 else f"beat {inside[0].index}")
        bars = sorted({x.bar for x in inside if x.bar >= 1}) if g.bars else []
        if bars:
            parts.append(f"bars {bars[0]}-{bars[-1]}" if len(bars) > 1 else f"bar {bars[0]}")
    parts.append(f"{t0:.3f}-{t1:.3f} s ({t1 - t0:.3f} s)")
    return ", ".join(parts)
