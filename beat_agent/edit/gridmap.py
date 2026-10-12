"""The beat grid of an edited track = the source grid mapped through the timeline.

Never re-detected: each output beat is a source beat that survived the edit,
moved by exactly the offset of the clip that carries it. Bars are renumbered
with the same rules as analysis (`analysis.numbering.number`), using each
beat's original position in its bar, so a cut that is not bar-aligned shows up
honestly as a partial bar.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from beat_agent import config
from beat_agent.analysis.numbering import number
from beat_agent.analysis.schema import BeatGrid, Confidence, Tempo, TempoSegment, TimeSignature
from beat_agent.edit.timeline import Timeline


@dataclass(frozen=True)
class SourceGrid:
    """Beats of the *source* audio in integer samples, plus per-beat bar position."""
    sr: int
    samples: np.ndarray        # int64, increasing
    positions: np.ndarray      # 0-based bar position, -1 if meter unknown
    regions: np.ndarray        # source beat-region id
    conf: np.ndarray
    seg_id: np.ndarray         # index into `segments` for each beat
    segments: tuple            # (kind, bpm_start, bpm_end) per source tempo segment
    beats_per_bar: int | None
    time_signature: TimeSignature
    base: BeatGrid             # the analysed grid (for source metadata / alternatives)
    octave: float = 1.0
    notes: tuple = ()

    @staticmethod
    def from_grid(g: BeatGrid) -> "SourceGrid":
        sr = int(g.source.get("samplerate", 44100))
        t = g.beat_times()
        samples = np.round(t * sr).astype(np.int64)
        bpb = g.time_signature.beats_per_bar
        pos = np.array([b.beat_in_bar - 1 if bpb else -1 for b in g.beats], dtype=np.int64)
        seg_id = np.zeros(len(t), dtype=np.int64)
        idx = np.array([b.index for b in g.beats])
        for k, s in enumerate(g.tempo.segments):
            seg_id[(idx >= s.start_beat) & (idx < s.end_beat)] = k
        segs = tuple((s.kind, s.bpm_start, s.bpm_end) for s in g.tempo.segments)
        return SourceGrid(sr, samples, pos, np.array([b.region for b in g.beats], dtype=np.int64),
                          np.array([b.confidence for b in g.beats]), seg_id, segs, bpb, g.time_signature, g)

    # --------------------------------------------------- grid corrections
    def shift_downbeat(self, k: int) -> "SourceGrid":
        if not self.beats_per_bar:
            raise ValueError("The meter is not determined, so there are no bar lines to shift.")
        return replace(self, positions=(self.positions - k) % self.beats_per_bar,
                       notes=self.notes + (f"bar lines shifted by {k:+d} beat(s)",))

    def set_beats_per_bar(self, m: int) -> "SourceGrid":
        if not len(self.samples):
            raise ValueError("There is no beat grid to re-bar.")
        db = np.flatnonzero(self.positions == 0)
        anchor = int(db[0]) if len(db) else 0
        pos = (np.arange(len(self.samples)) - anchor) % m
        unit = self.time_signature.beat_unit or "quarter"
        den = 8 if unit in ("eighth", "dotted_quarter") else 4
        num = 3 * m if unit == "dotted_quarter" else m
        ts = self.time_signature.model_copy(update={"beats_per_bar": m, "numerator": num, "denominator": den,
                                                    "confidence": 1.0, "candidates": {"user": m}})
        return replace(self, positions=pos, beats_per_bar=m, time_signature=ts,
                       notes=self.notes + (f"meter set to {num}/{den} by the user ({m} beats per bar)",))

    def set_octave(self, factor: float) -> "SourceGrid":
        """Correct the tracker's metrical level; the bar lines stay where they are.

        Downbeats come from their own detector, so a tempo-octave correction must
        not move or renumber bars: x2 inserts a beat halfway between neighbours
        (within a region) and doubles beats per bar; x0.5 keeps the beats on even
        bar positions and halves beats per bar. Halving is refused when bars have
        an odd number of beats (bar lines would fall between beats).
        """
        n = len(self.samples)
        if n < 2:
            raise ValueError("Too few beats to change the tempo octave.")
        m = self.beats_per_bar
        if factor == 2.0:
            s, r, c, sid, pos = [], [], [], [], []
            for i in range(n):
                p = int(self.positions[i])
                s.append(self.samples[i]); r.append(self.regions[i]); c.append(self.conf[i]); sid.append(self.seg_id[i])
                pos.append(2 * p if p >= 0 else -1)
                if i + 1 < n and self.regions[i + 1] == self.regions[i]:
                    s.append((self.samples[i] + self.samples[i + 1]) // 2)
                    r.append(self.regions[i]); c.append(0.0); sid.append(self.seg_id[i])
                    pos.append(2 * p + 1 if p >= 0 else -1)
            new_m = 2 * m if m else None
            s, r, c, sid, pos = map(np.asarray, (s, r, c, sid, pos))
        else:
            if m and m % 2:
                raise ValueError(f"Bars have {m} beats, so halving the tempo would put bar lines between beats. "
                                 "Tell me the meter instead (e.g. 'it's in 3/4'), or shift the downbeat first.")
            if m:
                sel = np.flatnonzero(self.positions % 2 == 0)
                pos = self.positions[sel] // 2
                new_m = m // 2
                if new_m < 2:
                    raise ValueError("Halving would leave one beat per bar; tell me the meter instead.")
            else:
                sel = np.arange(0, n, 2)
                pos, new_m = np.full(len(sel), -1), None
            s, r, c, sid = self.samples[sel], self.regions[sel], self.conf[sel], self.seg_id[sel]
        segs = tuple((k, b0 * factor, b1 * factor) for k, b0, b1 in self.segments)
        ts = self.time_signature
        if new_m:
            chain = ["eighth", "quarter", "half"]
            unit = ts.beat_unit or "quarter"
            if unit in chain:
                i = chain.index(unit) + (1 if factor == 0.5 else -1)
                unit = chain[i] if 0 <= i < len(chain) else unit
            elif unit == "dotted_quarter":
                unit = "dotted_half" if factor == 0.5 else "dotted_eighth"
            den = {"eighth": 8, "quarter": 4, "half": 2, "dotted_quarter": 8, "dotted_half": 4, "dotted_eighth": 16}.get(unit, 4)
            num = 3 * new_m if unit.startswith("dotted") else new_m
            ts = ts.model_copy(update={"beats_per_bar": new_m, "beat_unit": unit, "numerator": num, "denominator": den})
        return replace(self, samples=s, positions=np.asarray(pos), regions=r, conf=c, seg_id=sid, segments=segs,
                       beats_per_bar=new_m, time_signature=ts, octave=self.octave * factor,
                       notes=self.notes + (f"tempo octave x{factor:g} set by the user; bar lines unchanged",))


def map_grid(sg: SourceGrid, tl: Timeline, src_meta: dict, history: list) -> BeatGrid:
    sr = sg.sr
    out_s, pos, reg, conf, sid = [], [], [], [], []
    run = 0
    offs = tl.offsets()
    for i, (c, o) in enumerate(zip(tl.clips, offs)):
        if i > 0 and tl.clips[i - 1].src_end != c.src_start:
            run += 1
        lo, hi = np.searchsorted(sg.samples, [c.src_start, c.src_end], side="left")
        for j in range(lo, hi):
            out_s.append(o + int(sg.samples[j] - c.src_start))
            pos.append(int(sg.positions[j]))
            reg.append(run * 100000 + int(sg.regions[j]))
            conf.append(float(sg.conf[j]))
            sid.append(int(sg.seg_id[j]))
    out_s = np.array(out_s, dtype=np.int64)
    times = out_s / sr
    duration = tl.length / sr
    positions = np.array(pos, dtype=np.int64)
    beats, bars, downbeats = number(times, positions if sg.beats_per_bar else np.full(len(times), -1),
                                    np.array(reg, dtype=np.int64), np.array(conf), sg.beats_per_bar, duration)
    # tempo segments: maximal runs of beats from the same source segment and the same splice run
    segs, changes = [], []
    sid = np.array(sid)
    regs = np.array(reg)
    k = 0
    while k < len(beats):
        e = k
        while e + 1 < len(beats) and sid[e + 1] == sid[k] and regs[e + 1] // 100000 == regs[k] // 100000:
            e += 1
        kind, b0, b1 = sg.segments[sid[k]] if sg.segments else ("free", 0.0, 0.0)
        segs.append(TempoSegment(kind=kind, start_beat=beats[k].index, end_beat=beats[e].index + 1,
                                 start_time=beats[k].time, end_time=beats[e].time, bpm_start=b0, bpm_end=b1))
        if k > 0:
            prev = segs[-2]
            spliced = regs[k] // 100000 != regs[k - 1] // 100000
            if abs(b0 / max(1e-9, prev.bpm_end) - 1) >= config.TEMPO_CHANGE_MIN_RATIO:
                changes.append({"beat_index": beats[k].index, "time": beats[k].time, "bar": beats[k].bar,
                                "from_bpm": prev.bpm_end, "to_bpm": b0, "kind": "splice" if spliced else "step"})
        k = e + 1
    bpms = [s.bpm_start for s in segs] + [s.bpm_end for s in segs]
    if not beats:
        ttype, bpm = "none", None
    elif all(s.kind == "constant" for s in segs) and max(bpms) / max(1e-9, min(bpms)) - 1 < config.TEMPO_CHANGE_MIN_RATIO:
        ttype, bpm = "constant", float(np.average([s.bpm_start for s in segs],
                                                  weights=[max(1, s.end_beat - s.start_beat) for s in segs]))
    else:
        ttype = "variable"
        bpm = float(60.0 / np.median(np.diff(times))) if len(times) > 1 else None
    alts = [a.model_copy(update={"bpm": round(a.bpm * sg.octave, 3)}) for a in sg.base.tempo.alternatives]
    base_conf = sg.base.confidence
    meta = dict(src_meta)
    meta.update({"frames": tl.length, "duration_s": round(duration, 6)})
    return BeatGrid(
        source=meta, duration=duration,
        tempo=Tempo(bpm=None if bpm is None else round(bpm, 3), type=ttype, segments=segs, changes=changes,
                    alternatives=alts, octave_correction=sg.octave),
        time_signature=sg.time_signature, beats=beats, downbeats=downbeats, bars=bars, no_beat_regions=[],
        confidence=Confidence(overall=base_conf.overall, beat=base_conf.beat, downbeat=base_conf.downbeat,
                              coverage=base_conf.coverage, how="carried over from the source analysis; "
                              "edited grids are mapped, not re-detected"),
        messages=list(sg.notes), analysis={"derived_from": "source grid mapped through the edit timeline",
                                           "splices_s": [s / sr for s in tl.splices()]},
        edit_history=history)
