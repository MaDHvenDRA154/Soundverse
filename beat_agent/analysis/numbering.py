"""Assign beat indices, bars and beat_in_bar from bar positions.

Shared by analysis (positions from the downbeat decoder) and by the edit
executor (positions carried over from the source grid), so an edited grid is
numbered by exactly the same rules as a freshly analysed one.
"""
from __future__ import annotations

import numpy as np

from beat_agent.analysis.schema import Bar, Beat


def number(times: np.ndarray, positions: np.ndarray, regions: np.ndarray, confidences: np.ndarray,
           beats_per_bar: int | None, end_time: float) -> tuple[list[Beat], list[Bar], list[float]]:
    """times/positions/regions/confidences: one entry per beat, in time order.

    positions: 0-based bar position (0 = downbeat), or -1 if unknown.
    Returns (beats, bars, downbeat_times).
    """
    n = len(times)
    if n == 0:
        return [], [], []
    has_meter = beats_per_bar is not None and (positions >= 0).any()
    if has_meter:
        db = np.flatnonzero(positions == 0)
        anchor = int(db[0]) if len(db) else 0
    else:
        anchor = 0
    beats, bar_ids = [], []
    bar = 0
    for i in range(n):
        if has_meter:
            new_region = i > 0 and regions[i] != regions[i - 1]
            if positions[i] == 0:
                bar += 1
            elif new_region and bar > 0:
                bar += 1   # a region (or splice) that starts mid-bar opens a partial bar
            bib = int(positions[i]) + 1
        else:
            bib = 0
        bar_ids.append(bar)
        beats.append(Beat(time=float(times[i]), index=i - anchor, bar=bar if has_meter else 0,
                          beat_in_bar=bib, confidence=float(np.clip(confidences[i], 0, 1)), region=int(regions[i])))
    bars: list[Bar] = []
    downbeats = [float(times[i]) for i in range(n) if has_meter and positions[i] == 0]
    if has_meter:
        bar_ids = np.array(bar_ids)
        starts = np.flatnonzero(np.r_[True, bar_ids[1:] != bar_ids[:-1]])
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else n
            last_t = float(times[e - 1])
            if e < n and regions[e] == regions[e - 1]:
                end = float(times[e])
            else:
                period = float(np.median(np.diff(times[max(s, e - 4):e]))) if e - s >= 2 else (
                    float(np.median(np.diff(times))) if n > 1 else 0.0)
                end = min(end_time, last_t + period)
            pos = positions[s:e]
            complete = (e - s == beats_per_bar) and bool(np.all(pos == np.arange(beats_per_bar)))
            bars.append(Bar(bar=int(bar_ids[s]), start=float(times[s]), end=end, n_beats=int(e - s), complete=complete))
    return beats, bars, downbeats
