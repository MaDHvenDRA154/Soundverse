"""Audio -> Score JSON. Every note comes from a model or signal processing.

grid (Part 1) -> pick the bar range -> Demucs stems -> drums: band-split onset
detection (kick < 150 Hz, snare 150-5000 Hz, hats > 6 kHz) -> pitched stems:
Basic Pitch -> soft quantisation onto the grid -> Score.
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, find_peaks, sosfiltfilt

from beat_agent import config
from beat_agent.analysis.pipeline import analyze
from beat_agent.analysis.schema import BeatGrid
from beat_agent.io.audio import SourceInfo, open_source, read_range
from beat_agent.transcribe.score import Note, Patch, Score, Track


class RecreateError(ValueError):
    pass


_sep_lock = threading.Lock()
_separator = None


def separator():
    global _separator
    with _sep_lock:
        if _separator is None:
            from demucs.api import Separator
            _separator = Separator(model=config.DEMUCS_MODEL, device="cpu", progress=False)
        return _separator


# ------------------------------------------------------------- bar range
def choose_bars(grid: BeatGrid, bars: int | None, start_bar: int | None) -> tuple[int, int]:
    full = [b for b in grid.bars if b.bar >= 1 and b.complete]
    if not full:
        raise RecreateError("No complete bars were found, so there is no loop to recreate.")
    have = [b.bar for b in full]
    s = start_bar or have[0]
    if s not in have:
        raise RecreateError(f"Bar {s} is not a complete bar of this clip (complete bars: {have[0]}-{have[-1]}).")
    avail = [b for b in have if b >= s]
    run = 1
    while run < len(avail) and avail[run] == s + run:
        run += 1
    if bars is None:
        n = run if run <= config.RECREATE_MAX_BARS and len(have) <= config.RECREATE_MAX_BARS else min(config.RECREATE_DEFAULT_BARS, run)
    else:
        n = bars
    if n < 1 or n > config.RECREATE_MAX_BARS:
        raise RecreateError(f"A recreation is 1-{config.RECREATE_MAX_BARS} bars; {n} was asked for.")
    if n > run:
        raise RecreateError(f"Only {run} consecutive complete bars from bar {s} (asked for {n}).")
    return s, n


# ------------------------------------------------------------- beat mapping
class BeatMap:
    """seconds <-> beats through the actual grid (works for variable tempo)."""

    def __init__(self, grid: BeatGrid, t0: float, t1: float, loop_beats: float):
        t = grid.beat_times()
        k = np.arange(len(t), dtype=float)
        b0 = np.interp(t0, t, k)
        self.t, self.k = t, k - b0
        self.period = float(np.median(np.diff(t))) if len(t) > 1 else 0.5
        self.t0, self.loop_beats = t0, loop_beats

    def to_beats(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, float)
        b = np.interp(ts, self.t, self.k)
        b = np.where(ts < self.t[0], self.k[0] - (self.t[0] - ts) / self.period, b)
        return np.where(ts > self.t[-1], self.k[-1] + (ts - self.t[-1]) / self.period, b)

    def to_seconds(self, bs: np.ndarray) -> np.ndarray:
        bs = np.asarray(bs, float)
        s = np.interp(bs, self.k, self.t)
        s = np.where(bs < self.k[0], self.t[0] - (self.k[0] - bs) * self.period, s)
        return np.where(bs > self.k[-1], self.t[-1] + (bs - self.k[-1]) * self.period, s)


def quantise(b: np.ndarray, step: float, strength: float) -> np.ndarray:
    q = np.round(b / step) * step
    return b + strength * (q - b)


# ------------------------------------------------------------- drums
def _band_onsets(x: np.ndarray, sr: int, lo: float | None, hi: float | None):
    if lo and hi:
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    elif lo:
        sos = butter(4, lo, btype="high", fs=sr, output="sos")
    else:
        sos = butter(4, hi, btype="low", fs=sr, output="sos")
    y = sosfiltfilt(sos, x)
    hop = max(1, int(config.DRUM_HOP_S * sr))
    win = 2 * hop
    c = np.concatenate([[0.0], np.cumsum(y ** 2)])
    centres = np.arange(0, len(y), hop)
    a, b = np.clip(centres - win, 0, len(y)), np.clip(centres + win, 0, len(y))
    e = 10 * np.log10((c[b] - c[a]) / np.maximum(1, b - a) + 1e-12)
    lag = max(1, int(round(config.DRUM_RISE_S / config.DRUM_HOP_S)))
    rise = np.zeros_like(e)
    rise[lag:] = np.maximum(0.0, e[lag:] - e[:-lag])
    ref = np.percentile(rise, 99.5)
    if ref <= 0:
        return np.zeros(0), np.zeros(0)
    pk, props = find_peaks(rise, height=max(config.DRUM_MIN_RISE_DB, config.DRUM_REL_HEIGHT * ref),
                           distance=max(1, int(config.DRUM_MIN_SPACING_S / config.DRUM_HOP_S)))
    # a rise only counts if it reaches the loud part of this band: in log terms
    # the noise floor between hits also "rises" by 6 dB
    loud = np.percentile(e, 99)
    keep = e[np.minimum(pk, len(e) - 1)] >= loud - config.DRUM_MIN_LEVEL_BELOW_PEAK_DB
    pk, h = pk[keep], props["peak_heights"][keep]
    if not len(pk):
        return np.zeros(0), np.zeros(0)
    # (Tried: strength as linear energy gain instead of dB rise. On generated
    # hits it lost 3 of 12 kicks and added snare false positives; reverted.)
    # refine to ~1 ms with the steepest-rise detector on this band's signal
    from beat_agent.analysis.onsets import attack_times
    coarse = (pk - lag / 2.0) * hop / sr
    fine = attack_times(y, sr, coarse, config.DRUM_REFINE_S, config.DRUM_MIN_RISE_DB / 2)
    t = np.where(np.isfinite(fine), fine, coarse)
    return t, h / ref


def transcribe_drums(x: np.ndarray, sr: int) -> list[tuple[float, str, float]]:
    """(time_s, kind, strength 0..1) for each hit in the drum stem."""
    if np.sqrt(np.mean(x ** 2)) < 10 ** (config.STEM_SILENT_DB / 20):
        return []
    kt, ks = _band_onsets(x, sr, None, config.KICK_LOWPASS_HZ)
    st, ss = _band_onsets(x, sr, config.KICK_LOWPASS_HZ, config.SNARE_BAND_HI_HZ)
    ht, hs = _band_onsets(x, sr, min(config.HAT_BAND_LO_HZ, 0.4 * sr), None)
    hits = [(t, "kick", s) for t, s in zip(kt, ks)]
    tol = config.DRUM_COINCIDE_S
    for t, s in zip(st, ss):
        near_k = [k for k, (kt_, _, ks_) in enumerate(hits) if hits[k][1] == "kick" and abs(kt_ - t) < tol]
        # a kick's click also rises in the mid band; it is a snare only if the mid rise is clearly the stronger
        if near_k and s < config.SNARE_OVER_KICK * max(hits[k][2] for k in near_k):
            continue
        hits.append((t, "snare", s))
    for t, s in zip(ht, hs):
        if any(k in ("snare",) and abs(t0 - t) < tol for t0, k, _ in hits):
            continue   # a snare's noise reaches the hat band too
        hits.append((t, "hat", s))
    return sorted(hits)


# ------------------------------------------------------------- pitched
def transcribe_pitched(x: np.ndarray, sr: int, lo: int, hi: int, mono: bool) -> list[tuple[float, float, int, float]]:
    """(start_s, end_s, midi, amplitude) via Basic Pitch on one stem."""
    if np.sqrt(np.mean(x ** 2)) < 10 ** (config.STEM_SILENT_DB / 20):
        return []
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    import librosa
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "stem.wav"
        sf.write(p, x, sr, subtype="FLOAT")
        _, _, events = predict(str(p), ICASSP_2022_MODEL_PATH,
                               onset_threshold=config.BP_ONSET_THRESHOLD, frame_threshold=config.BP_FRAME_THRESHOLD,
                               minimum_note_length=config.BP_MIN_NOTE_MS,
                               minimum_frequency=float(librosa.midi_to_hz(lo)), maximum_frequency=float(librosa.midi_to_hz(hi)))
    notes = sorted((float(s), float(e), int(p_), float(a)) for s, e, p_, a, *_ in events)
    if mono and notes:
        out = []
        for n in notes:
            if out and n[0] < out[-1][1] - 1e-3:
                if n[3] > out[-1][3]:
                    out[-1] = (out[-1][0], n[0], out[-1][2], out[-1][3])  # cut the weaker previous note short
                    out.append(n)
                continue
            out.append(n)
        notes = [n for n in out if n[1] - n[0] > 0.02]
    return notes


def _centroid(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) if len(x) else np.zeros(1)
    f = np.fft.rfftfreq(len(x), 1 / sr) if len(x) else np.zeros(1)
    return float((spec * f).sum() / (spec.sum() + 1e-12))


# ------------------------------------------------------------- main
GM = {"kick": 36, "snare": 38, "hat": 42}


def recreate(path, bars: int | None = None, start_bar: int | None = None, src: SourceInfo | None = None,
             grid: BeatGrid | None = None, stems_dir: str | Path | None = None) -> Score:
    src = src or open_source(path)
    grid = grid or analyze(path, src)
    plan = plan_loop(grid, src, bars, start_bar)
    t0, t1, loop_beats, bpb, bm = plan["t0"], plan["t1"], plan["loop_beats"], plan["bpb"], plan["map"]
    pad = config.DEMUCS_CONTEXT_S
    sr = src.samplerate
    a0 = int(round((t0 - pad) * sr))
    seg = read_range(src, a0, int(round((t1 + pad) * sr)))
    if seg.shape[1] == 1:
        seg = np.repeat(seg, 2, axis=1)
    import torch
    sep = separator()
    if sep.samplerate != sr:
        import soxr
        seg = soxr.resample(seg, sr, sep.samplerate).astype(np.float32)
    ssr = sep.samplerate
    _, stems = sep.separate_tensor(torch.from_numpy(np.ascontiguousarray(seg[:, :2].T)), ssr)
    stems = {k: v.numpy().T.mean(axis=1) for k, v in stems.items()}
    off = (t0 - pad)   # source time of stem sample 0
    if stems_dir:
        Path(stems_dir).mkdir(parents=True, exist_ok=True)
        for k, v in stems.items():
            sf.write(Path(stems_dir) / f"{k}.wav", v, ssr, subtype="FLOAT")

    ternary = grid.time_signature.subdivision == "ternary" or grid.time_signature.beat_unit == "eighth"
    step = (1.0 / 3.0 if ternary else 0.25) if grid.time_signature.beat_unit != "eighth" else 0.5
    Q = config.QUANTIZE_STRENGTH
    ph = plan["phase"]   # quantise to the grid's beat phase, not to the loop start

    def place(start_src: float, end_src: float):
        b0, b1 = bm.to_beats(np.array([start_src, end_src])) - ph
        qb0 = float(quantise(np.array([b0]), step, Q)[0]) + ph
        qb1 = max(qb0 + step / 2, float(quantise(np.array([b1]), step, Q)[0]) + ph)
        return qb0, qb1

    tracks = []
    hits = transcribe_drums(stems["drums"], ssr)
    dn = []
    for t, kind, s in hits:
        ts = t + off
        b0, b1 = place(ts, ts + 0.1)
        if -step / 2 <= b0 < loop_beats - 1e-6:
            b0 = max(0.0, b0)
            s0 = float(bm.to_seconds(np.array([b0]))[0])
            dn.append(Note(start_beats=round(b0, 4), dur_beats=round(min(0.5, b1 - b0), 4), start_s=round(s0, 5),
                           dur_s=0.1, pitch=GM[kind], velocity=int(np.clip(40 + 87 * s, 1, 127)), drum=kind))
    if dn:
        tracks.append(Track(id="drums", name="Drums", role="drums", notes=dn))

    for stem, role, (lo, hi), mono, patch in (
            ("bass", "bass", config.BASS_RANGE, True, dict(osc="saw", cutoff_hz=700, resonance=2.0, decay_s=0.25, sustain=0.5, release_s=0.08)),
            ("other", "keys", config.KEYS_RANGE, False, dict(osc="fm", fm_ratio=1.0, fm_index=1.2, cutoff_hz=5000, decay_s=0.8, sustain=0.25, release_s=0.35)),
            ("vocals", "lead", config.LEAD_RANGE, True, dict(osc="square", cutoff_hz=2400, attack_s=0.02, sustain=0.7, release_s=0.15))):
        x = stems[stem]
        lo_i, hi_i = int(max(0, (t0 - off) * ssr)), int(min(len(x), (t1 - off) * ssr))
        notes = transcribe_pitched(x[lo_i:hi_i], ssr, lo, hi, mono)
        if not notes:
            continue
        if role == "bass":   # timbre: brightness of the stem sets the low-pass
            patch["cutoff_hz"] = float(np.clip(2.0 * _centroid(x[lo_i:hi_i], ssr), 200, 4000))
        out = []
        for s_rel, e_rel, p, amp in notes:
            b0, b1 = place(s_rel + t0, e_rel + t0)
            if b0 < -step / 2 or b0 >= loop_beats - 1e-6:
                continue
            b0 = max(0.0, b0)
            b1 = min(b1, loop_beats)
            s0, s1 = bm.to_seconds(np.array([b0, b1]))
            out.append(Note(start_beats=round(b0, 4), dur_beats=round(b1 - b0, 4), start_s=round(float(s0), 5),
                            dur_s=round(float(s1 - s0), 5), pitch=int(p), velocity=int(np.clip(30 + 97 * amp, 1, 127))))
        if out:
            tracks.append(Track(id=role, name={"bass": "Bass", "keys": "Keys", "lead": "Lead"}[role], role=role,
                                patch=Patch(**patch), notes=out))
    if not tracks:
        raise RecreateError("Nothing was transcribed from this clip (all stems silent or no onsets).")
    bpm = 60.0 * loop_beats / (t1 - t0)
    return Score(tempo_bpm=round(bpm, 3), analysed_tempo_bpm=round(bpm, 3), time_signature=plan["ts"],
                 beats_per_bar=bpb, bars=plan["bars"], loop_beats=loop_beats,
                 clip={"source": str(path), "start_s": round(t0, 5), "end_s": round(t1, 5), "bars": plan["bar_range"],
                       "mode": plan["mode"], "grid_bpm": grid.tempo.bpm, "grid_type": grid.tempo.type},
                 tracks=tracks,
                 provenance={"grid": "beat_agent.analysis (Beat This!)", "tempo_source": plan["tempo_source"],
                             "loop": plan["note"], "stems": config.DEMUCS_MODEL,
                             "drums": "band-split onset detection on the Demucs drum stem",
                             "pitched": "Basic Pitch (ONNX) on Demucs bass/other/vocals stems",
                             "quantise": {"step_beats": step, "strength": Q, "phase_beats": round(ph, 4)}, "llm": "none"})


class LinearMap:
    """Constant-tempo seconds <-> beats with beat 0 at t0 (whole-clip loops)."""

    def __init__(self, t0: float, period: float):
        self.t0, self.period = t0, period

    def to_beats(self, ts):
        return (np.asarray(ts, float) - self.t0) / self.period

    def to_seconds(self, bs):
        return self.t0 + np.asarray(bs, float) * self.period


def plan_loop(grid: BeatGrid, src: SourceInfo, bars: int | None, start_bar: int | None) -> dict:
    """Which stretch becomes the loop, and how seconds map to beats.

    whole-clip mode: a clip of <= RECREATE_MAX_BARS bars with no bar range asked
      for IS the loop. Loop = the whole file, a whole number of beats at the
      measured beat period, tempo set so the loop is exactly the clip length
      (seamless); notes are quantised to the grid's beat phase. Bar decoding is
      only used for labels here, so a wrong meter cannot cut the loop short.
    bars mode: longer tracks (or an explicit range) use complete bars of the grid.
    """
    dur = src.duration
    ts = grid.time_signature
    bpb = ts.beats_per_bar or 4
    period, tempo_source = None, "grid"
    t = grid.beat_times()
    if len(t) >= 4:
        period = float(np.median(np.diff(t)))
    else:
        import librosa
        from beat_agent.io.audio import load_analysis_mono
        y = load_analysis_mono(src)
        tempo = float(np.atleast_1d(librosa.feature.rhythm.tempo(y=y, sr=config.ANALYSIS_SR))[0])
        if tempo > 0:
            period, tempo_source = 60.0 / tempo, "onset-autocorrelation fallback (no beat grid)"
    if period is None:
        raise RecreateError("No tempo could be measured, so the clip cannot be looped. " + "; ".join(grid.messages))
    n_beats = dur / period
    whole = bars is None and start_bar is None and n_beats <= config.RECREATE_MAX_BARS * bpb + 0.5
    if whole:
        loop_beats = float(max(1, int(round(n_beats))))
        p_loop = dur / loop_beats
        phase = 0.0
        if len(t):
            ang = 2 * np.pi * (t / p_loop)
            phase = float((np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi)) % 1.0)
            if phase > 0.5:
                phase -= 1.0
        nb = max(1, int(round(loop_beats / bpb)))
        meter = (ts.numerator, ts.denominator) if ts.numerator else (4, 4)
        note = (f"whole clip as the loop: {loop_beats:.0f} beats of {p_loop * 1e3:.1f} ms = {dur:.3f} s"
                + ("" if ts.numerator else "; meter unknown, 4/4 assumed for bar labels only"))
        return {"mode": "whole_clip", "t0": 0.0, "t1": dur, "loop_beats": loop_beats, "bpb": bpb, "bars": min(nb, config.RECREATE_MAX_BARS),
                "bar_range": [1, nb], "map": LinearMap(0.0, p_loop), "phase": phase, "ts": meter,
                "tempo_source": tempo_source, "note": note}
    if not grid.bars:
        raise RecreateError("This clip is longer than 16 bars and has no bar grid, so a bar range cannot be chosen.")
    s_bar, n_bars = choose_bars(grid, bars, start_bar)
    t0, t1 = grid.bar(s_bar).start, grid.bar(s_bar + n_bars - 1).end
    lb = float(n_bars * bpb)
    return {"mode": "bars", "t0": t0, "t1": t1, "loop_beats": lb, "bpb": bpb, "bars": n_bars,
            "bar_range": [s_bar, s_bar + n_bars - 1], "map": BeatMap(grid, t0, t1, lb), "phase": 0.0,
            "ts": (ts.numerator, ts.denominator), "tempo_source": tempo_source,
            "note": f"bars {s_bar}-{s_bar + n_bars - 1} of the grid"}
