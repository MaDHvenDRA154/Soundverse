"""Frame-wise beat / downbeat activations from Beat This! (CPU).

We only use the network's activations; all decisions (regions, tempo segments,
grid fitting, meter) are made downstream in `grid.py`.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch

from beat_agent import config

FPS = 50  # Beat This! frame rate: hop 441 samples at 22.05 kHz


@dataclass
class Activations:
    beat: np.ndarray       # sigmoid probability per frame
    downbeat: np.ndarray
    fps: int = FPS

    def times(self) -> np.ndarray:
        return np.arange(len(self.beat)) / self.fps


_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from beat_this.inference import Spect2Frames
            from beat_this.preprocessing import LogMelSpect
            torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
            s2f = Spect2Frames(config.BEAT_THIS_CHECKPOINT, device="cpu")
            _model = (s2f, LogMelSpect(device="cpu"))
        return _model


def activations(y: np.ndarray, sr: int = config.ANALYSIS_SR) -> Activations:
    """y: mono float32 at ANALYSIS_SR."""
    if sr != config.ANALYSIS_SR:
        raise ValueError("resample to ANALYSIS_SR first")
    if len(y) < sr // 2:
        # shorter than the model's receptive field is useful for; pad with silence
        y = np.pad(y, (0, sr // 2 - len(y)))
    s2f, spect = _get_model()
    with torch.inference_mode():
        sp = spect(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)))
        b, d = s2f(sp)
    return Activations(beat=torch.sigmoid(b).numpy().astype(np.float64),
                       downbeat=torch.sigmoid(d).numpy().astype(np.float64))


def pick_peaks(act: np.ndarray, fps: int = FPS, threshold: float = config.PEAK_PROB_THRESHOLD,
               min_sep_s: float = config.PEAK_MIN_SEPARATION_S):
    """Local maxima above threshold. Optionally refined to sub-frame time by
    parabolic interpolation of the log-activation (config, off by default).
    Returns (times_s, heights)."""
    w = max(1, int(round(min_sep_s * fps)))
    n = len(act)
    if n < 3:
        return np.zeros(0), np.zeros(0)
    from scipy.ndimage import maximum_filter1d
    mx = maximum_filter1d(act, size=2 * w + 1, mode="constant")
    cand = np.flatnonzero((act == mx) & (act > threshold))
    # plateaus: keep the first frame of each run of equal maxima
    if len(cand) > 1:
        keep = np.ones(len(cand), bool)
        keep[1:] = np.diff(cand) > w
        cand = cand[keep]
    la = np.log(np.clip(act, 1e-6, 1.0))
    times, heights = [], []
    for i in cand:
        off = 0.0
        if config.PEAK_SUBFRAME_INTERPOLATION and 0 < i < n - 1:
            a, b, c = la[i - 1], la[i], la[i + 1]
            den = a - 2 * b + c
            if den < 0:
                off = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        times.append((i + off) / fps)
        heights.append(act[i])
    return np.array(times), np.array(heights)
