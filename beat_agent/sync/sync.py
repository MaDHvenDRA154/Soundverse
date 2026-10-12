"""Beat-match track B to master track A and render an overlap.

B is stretched with a Rubber Band time map that sends each of B's grid beats
onto the corresponding beat of A's grid, starting with B's chosen downbeat
on A's chosen downbeat. For two constant-tempo grids this is exactly a
global ratio + offset; for variable grids it stays locked where a single
ratio would drift.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soxr
from scipy.signal import find_peaks

from beat_agent import config
from beat_agent.analysis.onsets import lowband_envelope
from beat_agent.analysis.schema import BeatGrid
from beat_agent.io.audio import SourceInfo, open_writer, read_range
from beat_agent.sync.stretch import timemap_stretch


class SyncError(ValueError):
    """User-facing: why these two tracks cannot be synced as asked."""


@dataclass
class SyncPlan:
    a_bar: int
    b_bar: int
    a_start_s: float
    b_start_s: float
    bpm_a: float | None
    bpm_b: float | None
    octave_factor: float           # B's beats were taken at this factor to match A's level
    rate: float                    # B playback-rate factor (>1 = B sped up); bpm_a / (bpm_b*octave)
    stretch_ratio: float           # B duration multiplier = 1 / rate
    offset_s: float                # B time t maps to A time t + offset (at the anchor)
    n_mapped_beats: int
    overlap_s: float
    warnings: list = field(default_factory=list)


def _bar_start(grid: BeatGrid, bar: int, label: str) -> float:
    if grid.time_signature.beats_per_bar is None:
        raise SyncError(f"Track {label}: meter/downbeats not determined, so bar {bar} is undefined.")
    b = grid.bar(bar)
    if b is None:
        raise SyncError(f"Track {label} has bars 1-{grid.n_bars()}; bar {bar} does not exist.")
    return b.start


def plan(grid_a: BeatGrid, grid_b: BeatGrid, a_bar: int, b_bar: int = 1,
         overlap_s: float = config.SYNC_OVERLAP_S) -> tuple[SyncPlan, np.ndarray, np.ndarray]:
    """Returns (plan, B source beat times, A target beat times) for the mapped beats."""
    if not grid_a.beats or not grid_b.beats:
        raise SyncError("Both tracks need a detected beat grid to sync.")
    ta0 = _bar_start(grid_a, a_bar, "A")
    tb0 = _bar_start(grid_b, b_bar, "B")
    a = grid_a.beat_times()
    b = grid_b.beat_times()
    a = a[a >= ta0 - 1e-6]
    b = b[b >= tb0 - 1e-6]
    warnings = []
    bpm_a, bpm_b = grid_a.tempo.bpm, grid_b.tempo.bpm
    octave = 1.0
    if bpm_a and bpm_b:
        # take B at the metrical level closest to A's (a 70 BPM half-time grid
        # should beat-match a 140 BPM track beat-for-beat, not be stretched 2x)
        cands = {1.0: abs(np.log(bpm_a / bpm_b)), 2.0: abs(np.log(bpm_a / (2 * bpm_b))),
                 0.5: abs(np.log(bpm_a / (0.5 * bpm_b)))}
        octave = min(cands, key=cands.get)
        if octave == 2.0:
            b = np.sort(np.concatenate([b, (b[:-1] + b[1:]) / 2]))
            warnings.append("B's grid is at half A's tempo; B's beats were doubled to match A's level.")
        elif octave == 0.5:
            b = b[::2]
            warnings.append("B's grid is at double A's tempo; every other B beat was used.")
    a_end = ta0 + overlap_s
    # map beat k of B onto beat k of A, as far as both grids go, plus one beat past the window
    n_in = int(np.searchsorted(a, a_end)) + 1
    n = min(len(a), len(b), n_in)
    if n < 2:
        raise SyncError("Not enough beats after the chosen bars to compute a stretch.")
    if n < n_in and len(b) < n_in:
        warnings.append(f"B runs out of beats {a[n - 1] - ta0:.1f}s into the overlap.")
    if n < n_in and len(a) < n_in:
        warnings.append(f"A runs out of beats {a[n - 1] - ta0:.1f}s into the overlap.")
    a_m, b_m = a[:n], b[:n]
    rate = float((a_m[-1] - a_m[0]) / (b_m[-1] - b_m[0])) ** -1 if n >= 2 else 1.0
    if abs(np.log(rate)) > np.log(1.16):
        warnings.append(f"B must change speed by {100 * (rate - 1):+.1f}%; beyond about +-16% stretching artefacts become obvious.")
    if grid_a.tempo.type != "constant" or grid_b.tempo.type != "constant":
        warnings.append("At least one grid is variable-tempo; B is stretched beat-by-beat (time map), not by one ratio.")
    p = SyncPlan(a_bar=a_bar, b_bar=b_bar, a_start_s=float(ta0), b_start_s=float(tb0), bpm_a=bpm_a, bpm_b=bpm_b,
                 octave_factor=octave, rate=round(rate, 6), stretch_ratio=round(1 / rate, 6),
                 offset_s=round(float(ta0 - tb0), 6), n_mapped_beats=int(n),
                 overlap_s=float(min(overlap_s, a[-1] - ta0 if len(a) else 0.0, grid_a.duration - ta0)),
                 warnings=warnings)
    return p, b_m, a_m


def _match_channels(x: np.ndarray, ch: int) -> np.ndarray:
    if x.shape[1] == ch:
        return x
    if ch == 1:
        return x.mean(axis=1, keepdims=True)
    if x.shape[1] == 1:
        return np.repeat(x, ch, axis=1)
    m = x[:, :ch] if x.shape[1] > ch else np.hstack([x, np.repeat(x.mean(1, keepdims=True), ch - x.shape[1], 1)])
    return m


def render(src_a: SourceInfo, grid_a: BeatGrid, src_b: SourceInfo, grid_b: BeatGrid, out_dir,
           a_bar: int, b_bar: int = 1, overlap_s: float = config.SYNC_OVERLAP_S) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p, b_beats, a_beats = plan(grid_a, grid_b, a_bar, b_bar, overlap_s)
    sr = src_a.samplerate
    ch = src_a.channels
    a0 = int(round(p.a_start_s * sr))
    L = int(round(p.overlap_s * sr))
    seg_a = read_range(src_a, a0, a0 + L)

    # B excerpt from its first mapped beat to its last mapped beat (+ a beat of tail)
    srb = src_b.samplerate
    b_first, b_last = b_beats[0], b_beats[-1]
    b_tail = float(np.median(np.diff(b_beats))) if len(b_beats) > 1 else 0.5
    b0, b1 = int(round(b_first * srb)), int(round((b_last + b_tail) * srb))
    seg_b = read_range(src_b, b0, b1)
    if srb != sr:
        seg_b = soxr.resample(seg_b, srb, sr, quality="VHQ").astype(np.float32)
    seg_b = _match_channels(seg_b, ch)
    # time map in the excerpt's own sample frames (at A's rate)
    src_frames = np.round((b_beats - b_first) * sr).astype(np.int64)
    tgt_frames = np.round((a_beats - p.a_start_s) * sr).astype(np.int64)
    rate_tail = (tgt_frames[-1] - tgt_frames[-2]) / max(1, (src_frames[-1] - src_frames[-2])) if len(src_frames) > 1 else 1.0
    total_tgt = int(round(tgt_frames[-1] + (len(seg_b) - src_frames[-1]) * rate_tail))
    tm = [(int(s), int(t)) for s, t in zip(src_frames, tgt_frames)]
    tm.append((len(seg_b), total_tgt))
    stretched = timemap_stretch(seg_b, sr, tm, total_tgt)
    b_full = np.zeros((L, ch), np.float32)
    n_copy = min(L, len(stretched))
    b_full[:n_copy] = stretched[:n_copy]

    mix = seg_a + b_full
    peak = float(np.abs(mix).max()) if len(mix) else 0.0
    scale = 0.98 / peak if peak > 0.98 else 1.0
    mix *= scale
    paths = {}
    for name, data in (("overlap_mix", mix), ("a_segment", seg_a), ("b_stretched", b_full)):
        path = out_dir / f"{name}.wav"
        with open_writer(path, src_a, channels=ch) as w:
            w.write(data)
        paths[name] = str(path)

    metrics = kick_alignment(seg_a.mean(1), b_full.mean(1), sr, float(60.0 / grid_a.tempo.bpm) if grid_a.tempo.bpm else 0.5)
    result = {"plan": asdict(p), "mix_gain": round(scale, 4), "files": paths, "kick_alignment": metrics}
    (out_dir / "sync.json").write_text(json.dumps(result, indent=2))
    return result


def detect_kicks(x: np.ndarray, sr: int) -> np.ndarray:
    """Kick onset times in seconds.

    Low-band Hilbert envelope (no rectification ripple); candidate kicks are
    peaks of the envelope's *rise* over KICK_RISE_S, so the decay of one kick
    never counts as another. The onset is the sample where the envelope
    crosses KICK_ONSET_FRACTION of that kick's peak, linearly interpolated.
    """
    env = lowband_envelope(x, sr, config.KICK_LOWPASS_HZ, config.KICK_ENVELOPE_S)
    if not len(env) or env.max() <= 0:
        return np.zeros(0)
    # detection on a coarse envelope: kick and bass share the low band and
    # their interference makes the fine envelope oscillate; 15 ms averaging
    # removes that beating, leaving one rise per hit.
    w = max(1, int(config.KICK_DETECT_SMOOTH_S * sr))
    coarse = np.convolve(env, np.ones(w) / w, mode="same")
    lag = max(1, int(config.KICK_RISE_S * sr))
    rise = np.zeros_like(coarse)
    rise[lag:] = np.maximum(0.0, coarse[lag:] - coarse[:-lag])
    ref = np.percentile(rise, 99.9)
    if ref <= 0:
        return np.zeros(0)
    peaks, _ = find_peaks(rise, height=config.KICK_MIN_REL_LEVEL * ref, distance=int(config.KICK_MIN_SPACING_S * sr))
    pre = int(config.KICK_NEW_ONSET_LOOKBACK_S * sr)
    peaks = [p for p in peaks if p >= pre and coarse[min(len(coarse) - 1, p + lag)] >=
             config.KICK_NEW_ONSET_RATIO * coarse[max(0, p - pre):p].min()]
    onsets = []
    look = int(0.05 * sr)
    for pk in peaks:
        top = pk + int(np.argmax(env[pk:pk + look]))
        thr = config.KICK_ONSET_FRACTION * env[top]
        a = max(0, top - look)
        below = np.flatnonzero(env[a:top + 1] < thr)
        if not len(below):
            continue
        i = a + below[-1]
        y0, y1 = env[i], env[i + 1]
        frac = (thr - y0) / (y1 - y0) if y1 > y0 else 0.0
        onsets.append((i + frac) / sr)
    return np.unique(np.round(np.array(onsets), 6))


def kick_alignment(a: np.ndarray, b: np.ndarray, sr: int, beat_period: float) -> dict:
    ka, kb = detect_kicks(a, sr), detect_kicks(b, sr)
    if not len(ka) or not len(kb):
        return {"n_kicks_a": int(len(ka)), "n_kicks_b": int(len(kb)), "n_pairs": 0,
                "note": "no kicks detected in one of the tracks; alignment not measured"}
    max_d = config.KICK_PAIR_MAX_BEAT_FRACTION * beat_period
    offs = []
    for t in kb:
        j = int(np.argmin(np.abs(ka - t)))
        d = t - ka[j]
        if abs(d) <= max_d:
            offs.append(d)
    offs = np.array(offs)
    if not len(offs):
        return {"n_kicks_a": int(len(ka)), "n_kicks_b": int(len(kb)), "n_pairs": 0}
    ab = np.abs(offs) * 1e3
    return {"n_kicks_a": int(len(ka)), "n_kicks_b": int(len(kb)), "n_pairs": int(len(offs)),
            "median_abs_offset_ms": round(float(np.median(ab)), 3), "max_abs_offset_ms": round(float(ab.max()), 3),
            "p95_abs_offset_ms": round(float(np.percentile(ab, 95)), 3),
            "median_signed_offset_ms": round(float(np.median(offs) * 1e3), 3),
            "offsets_ms": [round(float(o) * 1e3, 3) for o in offs]}
