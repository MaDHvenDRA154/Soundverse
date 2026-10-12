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


def subdivision_ratio(beat_times: np.ndarray, flux: np.ndarray, fps: float) -> float | None:
    """min(E(1/3), E(2/3)) / E(1/2) over all beat intervals.

    Using the *min* of the two third-positions means swing (energy at 2/3 but
    not at 1/3) still reads as binary.
    """
    if len(beat_times) < 4:
        return None
    t0, t1 = beat_times[:-1], beat_times[1:]
    p = t1 - t0
    hw = 0.15 * np.median(p) / 3  # well inside a third of a beat
    e = lambda f: sample_max(flux, fps, t0 + f * p, hw).mean()
    half = e(1 / 2)
    third = min(e(1 / 3), e(2 / 3))
    return float(third / half) if half > 0 else None


def decode(region_beats: list[np.ndarray], downbeat_act: np.ndarray, act_fps: float,
           flux: np.ndarray, flux_fps: float, duration: float) -> MeterResult:
    all_beats = np.concatenate(region_beats) if region_beats else np.zeros(0)
    if len(all_beats) < config.MIN_BEATS_PER_REGION or duration < config.MIN_DURATION_FOR_METER_S:
        return MeterResult(None, [np.zeros(len(r), dtype=np.int64) for r in region_beats], {}, 0.0,
                           None, None, None, None, None)
    d_regions = [sample_max(downbeat_act, act_fps, r, 2.0 / act_fps) for r in region_beats]
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
    d = np.concatenate(d_regions)
    s = np.concatenate(states)
    is_db = s == 0
    hit = float((d[is_db] > 0.5).mean()) if is_db.any() else 0.0
    false = float((d[~is_db] > 0.5).mean()) if (~is_db).any() else 0.0
    conf = hit * (1.0 - false)

    ratio = None
    sub = "binary"
    if M in (2, 3, 4):
        ratio = subdivision_ratio(all_beats if len(region_beats) == 1 else region_beats[int(np.argmax([len(r) for r in region_beats]))], flux, flux_fps)
        if ratio is not None and ratio >= config.COMPOUND_RATIO_THRESHOLD:
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
