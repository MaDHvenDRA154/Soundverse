"""Signal features used to check the tracker against the audio itself."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft


def spectral_flux(y: np.ndarray, sr: int, hop_s: float, n_fft: int = 1024) -> tuple[np.ndarray, float]:
    """Half-wave rectified log-magnitude spectral flux. Returns (envelope, fps).

    Frame i is centred at i*hop (STFT with centre padding), the same convention
    as the tracker's frames.
    """
    hop = max(1, int(round(hop_s * sr)))
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    _, _, Z = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="even", padded=True)
    mag = np.log1p(100.0 * np.abs(Z))
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    return flux.astype(np.float64), sr / hop


def frame_rms_db(y: np.ndarray, sr: int, fps: float) -> np.ndarray:
    hop = int(round(sr / fps))
    n = int(np.ceil(len(y) / hop)) + 1
    pad = np.pad(y, (hop, hop + n * hop - len(y)))
    frames = pad[: (len(pad) // hop) * hop].reshape(-1, hop)
    e = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    # frame i covers [i*hop - hop, i*hop + hop) after the left pad -> centre on i*hop
    e = 0.5 * (e[:-1] + e[1:])[:n]
    return 20 * np.log10(e + 1e-10)


def sample_max(env: np.ndarray, fps: float, times: np.ndarray, half_window_s: float) -> np.ndarray:
    """Max of `env` within +-half_window around each time."""
    out = np.zeros(len(times))
    w = max(1, int(round(half_window_s * fps)))
    for i, t in enumerate(times):
        c = int(round(t * fps))
        a, b = max(0, c - w), min(len(env), c + w + 1)
        out[i] = env[a:b].max() if b > a else 0.0
    return out


def lowband_envelope(x: np.ndarray, sr: int, cutoff_hz: float, smooth_s: float) -> np.ndarray:
    """Hilbert envelope of the zero-phase low-passed signal (sample rate).

    Rectifying a 50 Hz kick leaves 100 Hz ripple that looks like new peaks;
    the analytic-signal magnitude does not. Zero-phase filtering keeps onset
    times unbiased.
    """
    from scipy.signal import hilbert
    sos = butter(4, cutoff_hz, btype="low", fs=sr, output="sos")
    low = sosfiltfilt(sos, x.astype(np.float64))
    env = np.abs(hilbert(low))
    n = max(1, int(smooth_s * sr))
    k = np.ones(n) / n
    return np.convolve(env, k, mode="same")
