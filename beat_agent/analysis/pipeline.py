"""analyze(path) -> BeatGrid. The only entry point the rest of the system uses."""
from __future__ import annotations

import time

import numpy as np

from beat_agent import config
from beat_agent.analysis import grid as gridmod
from beat_agent.analysis import meter as metermod
from beat_agent.analysis import tracker
from beat_agent.analysis.numbering import number
from beat_agent.analysis.onsets import sample_max, spectral_flux
from beat_agent.analysis.schema import (BeatGrid, Confidence, Region, Tempo, TempoAlternative,
                                        TempoSegment, TimeSignature)
from beat_agent.io.audio import SourceInfo, load_analysis_mono, open_source

CONFIDENCE_HOW = (
    "beat = median Beat This! activation at the grid beats; agreement = fraction of grid beats with "
    "a tracker peak within 35 ms; overall = beat x agreement. downbeat = (share of decoded downbeats "
    "with downbeat activation > 0.5) x (1 - share of other beats with activation > 0.5). "
    "coverage = fraction of the file inside beat regions."
)


def analyze(path, src: SourceInfo | None = None) -> BeatGrid:
    t_start = time.perf_counter()
    timings = {}
    src = src or open_source(path)
    y = load_analysis_mono(src)
    sr = config.ANALYSIS_SR
    duration = src.duration
    timings["load_s"] = time.perf_counter() - t_start

    t0 = time.perf_counter()
    act = tracker.activations(y, sr)
    timings["beat_this_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    flux, flux_fps = spectral_flux(y, sr, config.ONSET_HOP_S)
    peaks_t, peaks_h = tracker.pick_peaks(act.beat, act.fps)
    support = sample_max(flux, flux_fps, peaks_t, config.BEAT_SUPPORT_WINDOW_S)
    region_idx = gridmod.find_regions(peaks_t, peaks_h, support, duration)
    rgrids = [gridmod.fit_region(peaks_t[idx]) for idx in region_idx]
    rgrids = gridmod.merge_regions(rgrids, lambda ts: sample_max(flux, flux_fps, ts, config.BEAT_SUPPORT_WINDOW_S))
    # a fitted grid must not run past the file
    for rg in rgrids:
        keep = (rg.times >= 0) & (rg.times < duration)
        if not keep.all():
            _clip_region(rg, keep)
    rgrids = [rg for rg in rgrids if len(rg.times) >= gridmod.min_region_beats(duration)]
    region_times = [rg.times for rg in rgrids]

    mres = metermod.decode(region_times, act.downbeat, act.fps, flux, flux_fps, duration)
    timings["postprocess_s"] = time.perf_counter() - t0

    if rgrids:
        times = np.concatenate(region_times)
        regions = np.concatenate([np.full(len(r), k) for k, r in enumerate(region_times)])
        positions = np.concatenate(mres.states) if mres.beats_per_bar else np.full(len(times), -1)
        conf = sample_max(act.beat, act.fps, times, 2.0 / act.fps)
    else:
        times = regions = positions = conf = np.zeros(0)
    beats, bars, downbeats = number(times, positions.astype(int), regions.astype(int), conf,
                                    mres.beats_per_bar, duration)

    # ---- tempo segments (in global beat indices)
    offsets = np.cumsum([0] + [len(r) for r in region_times])
    anchor = -beats[0].index if beats else 0
    segments, changes = [], []
    for k, rg in enumerate(rgrids):
        for s in rg.segments:
            g0, g1 = offsets[k] + s.j0, offsets[k] + s.j1
            segments.append(TempoSegment(kind=s.kind, start_beat=int(g0 - anchor), end_beat=int(g1 - anchor),
                                         start_time=float(times[g0]), end_time=float(times[g1 - 1]),
                                         bpm_start=round(s.bpm_start, 3), bpm_end=round(s.bpm_end, 3),
                                         fit_residual_rms_ms=None if s.residual_rms_ms is None else round(s.residual_rms_ms, 2)))
        for j in rg.step_changes:
            g = offsets[k] + j
            before = next(s for s in rg.segments if s.j1 == j)
            after = next(s for s in rg.segments if s.j0 == j)
            changes.append({"beat_index": int(g - anchor), "time": float(times[g]), "bar": beats[g].bar,
                            "from_bpm": round(before.bpm_end, 3), "to_bpm": round(after.bpm_start, 3), "kind": "step"})
    for k in range(1, len(rgrids)):
        a, b = rgrids[k - 1].segments[-1], rgrids[k].segments[0]
        if a.bpm_end > 0 and abs(b.bpm_start / a.bpm_end - 1) >= config.TEMPO_CHANGE_MIN_RATIO:
            g = offsets[k]
            changes.append({"beat_index": int(g - anchor), "time": float(times[g]), "bar": beats[g].bar,
                            "from_bpm": round(a.bpm_end, 3), "to_bpm": round(b.bpm_start, 3), "kind": "across_gap"})

    if not beats:
        ttype, bpm = "none", None
    else:
        bpms = np.array([s.bpm_start for s in segments] + [s.bpm_end for s in segments])
        all_const = all(s.kind == "constant" for s in segments)
        if all_const and bpms.max() / bpms.min() - 1 < config.TEMPO_CHANGE_MIN_RATIO:
            ttype = "constant"
            w = np.array([s.end_beat - s.start_beat for s in segments], float)
            bpm = float(np.average([s.bpm_start for s in segments], weights=w))
        else:
            ttype = "variable"
            bpm = float(60.0 / np.median(np.concatenate([np.diff(r) for r in region_times])))

    alternatives = _octave_alternatives(region_times, flux, flux_fps, bpm) if bpm else []

    # ---- confidence
    if beats:
        beat_conf = float(np.median(conf))
        near = np.array([np.min(np.abs(peaks_t - t)) <= 0.035 if len(peaks_t) else False for t in times])
        agreement = float(near.mean())
        coverage = float(sum(min(duration, bar_end(rt)) - rt[0] for rt in region_times) / duration)
    else:
        beat_conf = agreement = coverage = 0.0
    confidence = Confidence(overall=round(beat_conf * agreement, 3), beat=round(beat_conf, 3),
                            downbeat=mres.confidence, coverage=round(min(1.0, coverage), 3), how=CONFIDENCE_HOW)

    no_beat = _no_beat_regions(region_times, duration)
    messages = []
    if not beats and len(peaks_t) >= 3:
        messages.append(f"The tracker found {len(peaks_t)} candidate beats, but no run of at least "
                        f"{gridmod.min_region_beats(duration)} regular, onset-backed beats, so no grid was produced "
                        "(too short, too irregular, or not really a pulse).")
    elif not beats:
        messages.append("No detectable beat in this audio: no grid was produced. "
                        "(Silence, noise, speech or beatless ambient material all look like this.)")
    if beats and mres.beats_per_bar is None:
        messages.append("Too little material to determine the meter; bars are not defined.")
    for r in no_beat:
        if r.end - r.start >= 2.0:
            messages.append(f"No detectable pulse from {r.start:.2f}s to {r.end:.2f}s; no beats emitted there.")
    if mres.beats_per_bar and mres.confidence < 0.5:
        messages.append(f"Downbeat/meter confidence is low ({mres.confidence:.2f}); bar numbers may be off.")

    ts = TimeSignature(numerator=mres.numerator, denominator=mres.denominator, beats_per_bar=mres.beats_per_bar,
                       beat_unit=mres.beat_unit, subdivision=mres.subdivision, subdivision_ratio=mres.subdivision_ratio,
                       confidence=mres.confidence, candidates={str(k): v for k, v in mres.loglik.items()})
    timings["total_s"] = time.perf_counter() - t_start
    return BeatGrid(
        source=src.to_dict(), duration=duration,
        tempo=Tempo(bpm=None if bpm is None else round(bpm, 3), type=ttype, segments=segments,
                    changes=changes, alternatives=alternatives),
        time_signature=ts, beats=beats, downbeats=downbeats, bars=bars, no_beat_regions=no_beat,
        confidence=confidence, messages=messages,
        analysis={"tracker": f"beat_this:{config.BEAT_THIS_CHECKPOINT}", "n_tracker_peaks": int(len(peaks_t)),
                  "n_inserted_beats": int(sum(rg.inserted.sum() for rg in rgrids)),
                  "timings": {k: round(v, 3) for k, v in timings.items()}},
    )


def bar_end(rt: np.ndarray) -> float:
    return float(rt[-1] + (rt[-1] - rt[-2] if len(rt) > 1 else 0.0))


def _clip_region(rg: gridmod.RegionGrid, keep: np.ndarray) -> None:
    first, last = np.argmax(keep), len(keep) - np.argmax(keep[::-1])
    rg.times, rg.inserted = rg.times[first:last], rg.inserted[first:last]
    segs = []
    for s in rg.segments:
        a, b = max(s.j0, first), min(s.j1, last)
        if b > a:
            s.j0, s.j1, s.times = a - first, b - first, s.times[a - s.j0:b - s.j0]
            segs.append(s)
    rg.segments = segs
    rg.step_changes = [j - first for j in rg.step_changes if first < j < last]


def _no_beat_regions(region_times: list[np.ndarray], duration: float) -> list[Region]:
    if not region_times:
        return [Region(start=0.0, end=duration)]
    out, cursor = [], 0.0
    for rt in region_times:
        period = float(np.median(np.diff(rt))) if len(rt) > 1 else 0.5
        if rt[0] - cursor > 2 * period:
            out.append(Region(start=cursor, end=float(rt[0])))
        cursor = bar_end(rt)
    period = float(np.median(np.diff(region_times[-1]))) if len(region_times[-1]) > 1 else 0.5
    if duration - cursor > 2 * period:
        out.append(Region(start=cursor, end=duration))
    return out


def _octave_alternatives(region_times, flux, fps, bpm) -> list[TempoAlternative]:
    """Evidence for the neighbouring tempo octaves, relative to the tracked level.

    x2:   onset strength at the half-beat points / at the beats. Near 1 means
          the half-beats are as strong as the beats (double time plausible);
          near 0 means nothing happens between beats.
    x0.5: onset strength at the weaker / stronger of the two alternating beat
          phases. Near 1 means alternate beats are equally strong (no reason to
          halve); a low value means a strong-weak pattern (half time plausible,
          e.g. a backbeat-heavy half-time feel).
    The tracked level is never switched automatically (see DECISIONS.md).
    """
    if not region_times:
        return []
    def s(ts):
        return float(sample_max(flux, fps, ts, 0.02).mean()) if len(ts) else 0.0
    one = np.concatenate(region_times)
    on_beat = s(one) + 1e-9
    out = [TempoAlternative(factor=1.0, bpm=round(bpm, 3), onset_support=1.0)]
    for f in config.TEMPO_OCTAVE_FACTORS:
        if f == 2.0:
            mids = np.concatenate([(r[:-1] + r[1:]) / 2 for r in region_times])
            v = s(mids) / on_beat
        elif f == 0.5:
            p0 = s(np.concatenate([r[0::2] for r in region_times]))
            p1 = s(np.concatenate([r[1::2] for r in region_times]))
            v = min(p0, p1) / (max(p0, p1) + 1e-9)
        else:
            continue
        out.append(TempoAlternative(factor=f, bpm=round(bpm * f, 3), onset_support=round(v, 3)))
    return out
