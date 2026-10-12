"""Synthetic music with known ground truth, generated in code.

Used by the generalisation / robustness tests and by `make eval` when no test
pack is present. Ground truth is exact because we place every event ourselves.
Nothing here is imported by beat_agent/.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# beats_per_bar, subdivisions per beat for each meter we generate
METERS = {
    "2/4": (2, 2), "3/4": (3, 2), "4/4": (4, 2), "6/8": (2, 3), "12/8": (4, 3),
}


@dataclass
class Truth:
    beats: np.ndarray
    downbeats: np.ndarray
    meter: str
    beats_per_bar: int
    subdivision: int
    # (first_beat_index, last_beat_index_exclusive, start_bpm, end_bpm)
    tempo_segments: list = field(default_factory=list)
    beat_start_s: float = 0.0        # no beats before this (intro / offset)
    duration_s: float = 0.0
    kind: str = "music"              # music | silence | noise

    @property
    def is_constant(self) -> bool:
        return len(self.tempo_segments) == 1 and self.tempo_segments[0][2] == self.tempo_segments[0][3]

    def to_dict(self) -> dict:
        return {"beats": self.beats.tolist(), "downbeats": self.downbeats.tolist(), "meter": self.meter,
                "beats_per_bar": self.beats_per_bar, "subdivision": self.subdivision,
                "tempo_segments": self.tempo_segments, "beat_start_s": self.beat_start_s,
                "duration_s": self.duration_s, "kind": self.kind}


# --------------------------------------------------------------------------- sounds
def _env(n, sr, decay):
    """Exponential decay that also fades to exactly zero over the last 10 ms:
    a sample cut off mid-decay ends in a click (found when the drum transcriber
    reported 'snares' 350 ms after every generated kick)."""
    e = np.exp(-np.arange(n) / (decay * sr))
    f = min(n, int(0.010 * sr))
    e[n - f:] *= np.linspace(1.0, 0.0, f)
    return e


def _kick(sr, rng):
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    f = 48 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / sr
    click = rng.normal(0, 1, n) * _env(n, sr, 0.002) * 0.3
    return (np.sin(ph) * _env(n, sr, 0.18) + click).astype(np.float32)


def _band(x, sr, lo=None, hi=None):
    if lo and hi:
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    elif lo:
        sos = butter(4, lo, btype="high", fs=sr, output="sos")
    else:
        sos = butter(4, hi, btype="low", fs=sr, output="sos")
    return sosfilt(sos, x)


def _snare(sr, rng):
    n = int(0.25 * sr)
    nyq_hi = min(6000, 0.45 * sr)
    noise = _band(rng.normal(0, 1, n), sr, 900, nyq_hi) * _env(n, sr, 0.07)
    tone = np.sin(2 * np.pi * 185 * np.arange(n) / sr) * _env(n, sr, 0.04)
    return (0.6 * noise / (np.abs(noise).max() + 1e-9) + 0.5 * tone).astype(np.float32)


def _hat(sr, rng):
    n = int(0.06 * sr)
    lo = min(7000, 0.35 * sr)
    x = _band(rng.normal(0, 1, n), sr, lo=lo) * _env(n, sr, 0.012)
    return (0.25 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def _crash(sr, rng):
    n = int(1.2 * sr)
    lo = min(4500, 0.3 * sr)
    x = _band(rng.normal(0, 1, n), sr, lo=lo) * _env(n, sr, 0.35)
    return (0.3 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def _midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _tone(sr, n, midi, kind="saw", attack=0.01, release=0.05):
    t = np.arange(n) / sr
    f = _midi_hz(midi)
    if kind == "saw":
        x = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 8) if f * k < 0.45 * sr)
    else:
        x = np.sin(2 * np.pi * f * t)
    a = np.minimum(1, t / attack)
    r = np.minimum(1, (n / sr - t) / release).clip(0, 1)
    return (x * a * r).astype(np.float32)


def _add(out, x, pos, gain=1.0):
    if pos >= len(out):
        return
    n = min(len(x), len(out) - pos)
    if pos < 0:
        return
    out[pos:pos + n] += gain * x[:n]


# --------------------------------------------------------------------------- tempo
def beat_times_from_segments(segments, start_s):
    """segments: list of (n_beats, bpm_start, bpm_end); bpm interpolated per beat."""
    times, seg_info, t, k = [], [], start_s, 0
    for n, b0, b1 in segments:
        seg_info.append((k, k + n, float(b0), float(b1)))
        for i in range(n):
            bpm = b0 + (b1 - b0) * (i / max(1, n - 1)) if b1 != b0 else b0
            times.append(t)
            t += 60.0 / bpm
            k += 1
    return np.array(times), seg_info, t


# --------------------------------------------------------------------------- tracks
def make_track(*, sr=44100, segments=((64, 120, 120),), meter="4/4", start_s=0.5, swing=0.0,
               intro_pad_s=0.0, tail_s=1.5, seed=0, four_on_floor=False, channels=2):
    """Render a drum + bass + pad track. Returns (audio (n, ch) float32, Truth)."""
    rng = np.random.default_rng(seed)
    bpb, sub = METERS[meter]
    t0 = start_s + intro_pad_s
    beats, seg_info, t_end = beat_times_from_segments(segments, t0)
    n_beats = len(beats)
    # make the track end on a bar line
    n_beats -= n_beats % bpb
    beats = beats[:n_beats]
    dur = (beats[-1] + 60.0 / segments[-1][2] if n_beats else t0) + tail_s
    n = int(dur * sr)
    out = np.zeros(n, dtype=np.float32)
    kick, snare, hat, crash = _kick(sr, rng), _snare(sr, rng), _hat(sr, rng), _crash(sr, rng)
    # chord progression: one chord per bar so harmony marks the downbeat
    roots = [45, 41, 48, 43]  # A, F, C, G (as MIDI)
    periods = np.diff(np.append(beats, beats[-1] + 60.0 / segments[-1][2]))

    for i, bt in enumerate(beats):
        pos_in_bar = i % bpb
        bar = i // bpb
        p = int(round(bt * sr))
        period = periods[i]
        # drums
        if four_on_floor or pos_in_bar == 0 or (bpb == 4 and pos_in_bar == 2 and sub == 2):
            _add(out, kick, p, 0.9)
        if (bpb == 4 and pos_in_bar in (1, 3)) or (bpb in (2, 3) and pos_in_bar == bpb - 1 and bpb != 1):
            _add(out, snare, p, 0.6)
        if bpb == 3 and pos_in_bar == 1:
            _add(out, snare, p, 0.35)
        for s in range(sub):
            frac = s / sub
            if sub == 2 and s == 1 and swing:
                frac = 0.5 + swing * (2 / 3 - 0.5)
            _add(out, hat, int(round((bt + frac * period) * sr)), 0.9 if s == 0 else 0.6)
        if pos_in_bar == 0 and bar % 4 == 0:
            _add(out, crash, p, 0.7)
        # bass: root on every beat, octave on the last beat of the bar
        root = roots[bar % len(roots)]
        bl = int(period * sr * 0.9)
        _add(out, _tone(sr, bl, root + (12 if pos_in_bar == bpb - 1 else 0)), p, 0.18)
        # pad chord held over the bar, re-struck at each downbeat
        if pos_in_bar == 0:
            bar_len = int(sum(periods[i:i + bpb]) * sr)
            for iv in (12, 16, 19):
                _add(out, _tone(sr, bar_len, root + iv, "sine", attack=0.02, release=0.08), p, 0.07)

    if intro_pad_s > 0:
        # sustained chords with slow attacks: harmony, no pulse
        seg_len = int(intro_pad_s * sr / 3)
        for j in range(3):
            for iv in (12, 16, 19, 24):
                _add(out, _tone(sr, seg_len, roots[j] + iv, "sine", attack=1.5, release=1.5),
                     int(start_s * sr) + j * seg_len, 0.06)

    out /= max(1e-9, np.abs(out).max() / 0.8)
    down = beats[::bpb]
    truth = Truth(beats=beats, downbeats=down, meter=meter, beats_per_bar=bpb, subdivision=sub,
                  tempo_segments=seg_info, beat_start_s=float(beats[0]) if n_beats else dur, duration_s=n / sr)
    audio = np.repeat(out[:, None], channels, axis=1)
    if channels >= 2:  # tiny decorrelation so stereo is real stereo
        audio[:, 1] = np.roll(audio[:, 1], 3)
    return audio.astype(np.float32), truth


def make_silence(sr=44100, dur=20.0, channels=2):
    return np.zeros((int(sr * dur), channels), np.float32), Truth(np.array([]), np.array([]), "-", 0, 0, [], dur, dur, "silence")


def make_noise(sr=44100, dur=20.0, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    x = (0.1 * rng.normal(0, 1, (int(sr * dur), channels))).astype(np.float32)
    return x, Truth(np.array([]), np.array([]), "-", 0, 0, [], dur, dur, "noise")


def write(path, audio, sr, subtype="PCM_16"):
    sf.write(str(path), audio, sr, subtype=subtype)
    return path


def random_case(seed: int, *, min_bpm=70, max_bpm=175, meters=("3/4", "4/4", "6/8"), n_bars=24):
    """A random tempo/meter/offset case for the generalisation test."""
    rng = np.random.default_rng(seed)
    meter = meters[rng.integers(len(meters))]
    bpb = METERS[meter][0]
    # For compound meters the beat is the dotted quarter; keep its tempo in a
    # range where the pulse is unambiguous.
    lo, hi = (min_bpm, max_bpm) if METERS[meter][1] == 2 else (max(min_bpm, 50), min(max_bpm, 110))
    bpm = float(np.round(rng.uniform(lo, hi), 2))
    start = float(np.round(rng.uniform(0.0, 3.0), 3))
    return dict(segments=((n_bars * bpb, bpm, bpm),), meter=meter, start_s=start, seed=seed)
