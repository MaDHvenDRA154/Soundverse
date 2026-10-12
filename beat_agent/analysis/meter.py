"""Meter and downbeat decoding on top of a beat grid.

For each candidate beats-per-bar M we run Viterbi over bar positions
(0..M-1), where position 0 emits the network's downbeat probability. The bar
may restart early with a small probability so a dropped beat in a live
recording does not shift every later bar. M is chosen by total likelihood.
Compound vs simple (6/8 vs 2/4, 12/8 vs 4/4) comes from the subdivision:
onset energy at beat thirds vs beat halves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beat_agent import config
from beat_agent.analysis.onsets import sample_max


@dataclass
class MeterResult:
    beats_per_bar: int | None
    states: list[np.ndarray]          # per region: bar position (0 = downbeat) per beat
    loglik: dict
    confidence: float
    numerator: int | None
    denominator: int | None
    beat_unit: str | None
    subdivision: str | None
    subdivision_ratio: float | None


def _viterbi(d: np.ndarray, M: int, eps: float) -> tuple[np.ndarray, float]:
    d = np.clip(d, 0.01, 0.99)
    le = np.stack([np.log(d)] + [np.log(1 - d)] * (M - 1), axis=1)  # (n, M) emission per position
    n = len(d)
    lstay, lrestart = np.log(1 - eps), np.log(eps)
    delta = np.full(M, -np.log(M)) + le[0]
    back = np.zeros((n, M), dtype=np.int64)
    for i in range(1, n):
        new = np.full(M, -np.inf)
        arg = np.zeros(M, dtype=np.int64)
        for s in range(M):
            nxt = (s + 1) % M
            v = delta[s] + (0.0 if s == M - 1 else lstay)
            if v > new[nxt]:
                new[nxt], arg[nxt] = v, s
            if s != M - 1:  # early restart of the bar
                v = delta[s] + lrestart
                if v > new[0]:
                    new[0], arg[0] = v, s
        delta = new + le[i]
        back[i] = arg
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    ll = float(delta[path[-1]])
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path, ll


def subdivision_evidence(beat_times: np.ndarray, flux: np.ndarray, fps: float) -> tuple[float, float] | None:
    """(triplet, duplet) onset energy inside the beat, relative to on-beat energy.

    triplet = min(E(1/3), E(2/3)) / E(0), duplet = E(1/2) / E(0), measured on
    high-band flux (hats, shakers, strums carry the subdivision; bass and pads
    only blur it). Using the *min* of the two third-positions means swing
    (energy at 2/3 but not at 1/3) reads as binary, and normalising both by the
    on-beat energy avoids dividing by an empty half-beat.
    """
    if len(beat_times) < 4:
        return None
    t0, t1 = beat_times[:-1], beat_times[1:]
    p = t1 - t0
    hw = 0.15 * np.median(p) / 3  # well inside a third of a beat
    e = lambda f: sample_max(flux, fps, t0 + f * p, hw).mean()
    on = e(0.0)
    if on <= 0:
        return None
    return float(min(e(1 / 3), e(2 / 3)) / on), float(e(1 / 2) / on)


def is_ternary(ev: tuple[float, float] | None) -> bool:
    if ev is None:
        return False
    trip, dup = ev
    return trip >= config.SUBDIVISION_MIN_REL and trip >= config.COMPOUND_RATIO_THRESHOLD * dup


def chroma_evidence(y: np.ndarray, sr: int, beat_times: np.ndarray) -> tuple[np.ndarray, bool]:
    """(per-beat harmonic-change score in [0, 1], whether harmony is informative)."""
    raw = chroma_change(y, sr, beat_times, normalise=False)
    ref = float(np.percentile(raw, 95)) if len(raw) else 0.0
    if ref < config.CHROMA_MIN_CHANGE:
        return np.full(len(raw), 0.5), False   # static harmony: no evidence either way
    return np.clip(raw / ref, 0.0, 1.0), True


def chroma_change(y: np.ndarray, sr: int, beat_times: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Harmonic change entering each beat: 1 - cosine(chroma of this beat, previous beat).

    Chords tend to change on bar lines, so this is independent downbeat
    evidence for when the network's downbeat output is ambiguous.
    """
    import librosa
    n = len(beat_times)
    if n < 2:
        return np.zeros(n)
    hop = 512
    C = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    idx = np.clip(np.round(np.append(beat_times, beat_times[-1] + np.median(np.diff(beat_times))) * sr / hop).astype(int),
                  0, C.shape[1])
    pb = np.zeros((n, 12))
    for i in range(n):
        a, b = idx[i], max(idx[i + 1], idx[i] + 1)
        if a < C.shape[1]:
            pb[i] = C[:, a:b].mean(axis=1)
    pb /= np.linalg.norm(pb, axis=1, keepdims=True) + 1e-9
    nov = np.r_[0.0, 1.0 - (pb[1:] * pb[:-1]).sum(axis=1)]
    if not normalise:
        return nov
    ref = np.percentile(nov, 95)
    return np.clip(nov / ref, 0.0, 1.0) if ref > 0 else np.zeros(n)


def decode(region_beats: list[np.ndarray], downbeat_act: np.ndarray, act_fps: float,
           flux: np.ndarray, flux_fps: float, duration: float,
           y: np.ndarray | None = None, sr: int | None = None) -> MeterResult:
    """flux: high-band onset envelope (for the subdivision test).
    y/sr: analysis audio, for the harmonic-change downbeat cue (optional)."""
    all_beats = np.concatenate(region_beats) if region_beats else np.zeros(0)
    if len(all_beats) < config.MIN_BEATS_PER_REGION or duration < config.MIN_DURATION_FOR_METER_S:
        return MeterResult(None, [np.zeros(len(r), dtype=np.int64) for r in region_beats], {}, 0.0,
                           None, None, None, None, None)
    d_net = [sample_max(downbeat_act, act_fps, r, 2.0 / act_fps) for r in region_beats]
    d_regions = []
    for r, d in zip(region_beats, d_net):
        if y is not None and len(r) >= 2:
            h, informative = chroma_evidence(y, sr, r)
        else:
            h, informative = None, False
        if informative:
            w = config.DOWNBEAT_NETWORK_WEIGHT
            d = np.exp(w * np.log(np.clip(d, 0.01, 0.99)) + (1 - w) * np.log(np.clip(h, 0.01, 0.99)))
        d_regions.append(d)
    loglik, paths = {}, {}
    for M in config.METER_CANDIDATES:
        ll, ps = 0.0, []
        for d in d_regions:
            path, l = _viterbi(d, M, config.DOWNBEAT_RESTART_PROB)
            ps.append(path)
            ll += l
        n_bars = len(all_beats) / M
        loglik[M] = ll + config.METER_PRIOR_LOGBONUS.get(M, 0.0) * n_bars
        paths[M] = ps
    M = max(loglik, key=loglik.get)
    states = paths[M]

    # confidence: decoded downbeats should carry high downbeat activation and
    # the other beats low; product of the two hit rates.
    d = np.concatenate(d_net)
    s = np.concatenate(states)
    is_db = s == 0
    hit = float((d[is_db] > 0.5).mean()) if is_db.any() else 0.0
    false = float((d[~is_db] > 0.5).mean()) if (~is_db).any() else 0.0
    conf = hit * (1.0 - false)

    ratio = None
    sub = "binary"
    if M in (2, 3, 4):
        longest = region_beats[int(np.argmax([len(r) for r in region_beats]))]
        ev = subdivision_evidence(longest, flux, flux_fps)
        if ev is not None:
            ratio = ev[0] / max(ev[1], 1e-3)
        if is_ternary(ev):
            sub = "ternary"
    if M == 6:
        num, den, unit, sub = 6, 8, "eighth", "binary"   # beats are eighths grouped 3+3
    elif sub == "ternary":
        num, den, unit = 3 * M, 8, "dotted_quarter"
    else:
        num, den, unit = M, 4, "quarter"
    norm = {int(k): round(v / max(1, len(all_beats)), 4) for k, v in loglik.items()}
    return MeterResult(M, states, norm, round(conf, 3), num, den, unit, sub,
                       None if ratio is None else round(ratio, 3))
