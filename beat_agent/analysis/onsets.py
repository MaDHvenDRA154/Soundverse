"""Signal features used to check the tracker against the audio itself."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft


def spectral_flux(y: np.ndarray, sr: int, hop_s: float, n_fft: int = 1024,
                  fmin: float = 0.0) -> tuple[np.ndarray, float]:
    """Half-wave rectified log-magnitude spectral flux. Returns (envelope, fps).

    Frame i is centred at i*hop (STFT with centre padding), the same convention
    as the tracker's frames. fmin restricts the flux to bins above fmin.
    """
    hop = max(1, int(round(hop_s * sr)))
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    f, _, Z = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="even", padded=True)
    if fmin > 0:
        Z = Z[f >= fmin]
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


def attack_times(y: np.ndarray, sr: int, times: np.ndarray, half_window_s: float,
                 min_rise_db: float, win_s: float = 0.004, hop_s: float = 0.001,
                 rise_s: float = 0.003, loud_db: float = 6.0) -> np.ndarray:
    """Time of the steepest log-energy rise near each time (NaN if no attack).

    A 4 ms energy window at 1 ms hop resolves a drum attack to ~1 ms, far
    finer than the tracker's 20 ms frames. Returns NaN where the steepest rise
    within +-half_window is below min_rise_db (no percussive event there).
    """
    hop = max(1, int(round(hop_s * sr)))
    hop_s = hop / sr          # the hop actually used (22 samples at 22.05 kHz is 0.9977 ms, not 1 ms)
    win = max(2, int(round(win_s * sr)))
    c = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    centres = np.arange(0, len(y), hop)
    a = np.clip(centres - win // 2, 0, len(y))
    b = np.clip(centres + win // 2, 0, len(y))
    e = (c[b] - c[a]) / np.maximum(1, b - a)
    L = 10 * np.log10(e + 1e-10)
    lag = max(1, int(round(rise_s / hop_s)))
    rise = np.full(len(L), -np.inf)
    rise[lag:] = L[lag:] - L[:-lag]
    out = np.full(len(times), np.nan)
    w = int(round(half_window_s / hop_s))
    for i, t in enumerate(times):
        k = int(round(t / hop_s))
        lo, hi = max(lag, k - w), min(len(L), k + w + 1)
        if hi <= lo:
            continue
        # In log terms a quiet hat rising out of silence outscores a kick rising
        # over a bass note; only rises that reach the loud part of the window
        # (within 6 dB of its peak level, looking a few ms ahead) qualify.
        ahead = np.maximum.accumulate(L[lo:hi + lag][::-1])[::-1][:hi - lo] if hi + lag <= len(L) else L[lo:hi]
        loud = ahead >= L[lo:min(len(L), hi + lag)].max() - loud_db
        cand = np.where(loud, rise[lo:hi], -np.inf)
        j = lo + int(np.argmax(cand))
        if np.isfinite(cand.max()) and rise[j] >= min_rise_db:
            out[i] = (j - lag / 2.0) * hop_s
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
