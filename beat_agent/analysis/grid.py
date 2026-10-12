"""From tracker peaks to a beat grid: regions, repair, tempo segments, fitting.

Pipeline per track:
  peaks -> regions (runs with a real, regular pulse; edges trimmed where no
  onset supports the beat) -> repair (fill a missed beat / drop a doubled one)
  -> tempo model per region, tried in order of simplicity:
       constant (one straight line, beat time vs index)
       steps    (piecewise constant; every piece must pass the constant test)
       ramp     (quadratic: steady accelerando / ritardando)
       free     (the tracked beats themselves, lightly smoothed)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

from beat_agent import config


@dataclass
class Segment:
    kind: str                      # constant | ramp | free
    j0: int                        # first beat (region-local index), inclusive
    j1: int                        # last beat, exclusive
    times: np.ndarray              # grid times for beats j0..j1-1
    bpm_start: float
    bpm_end: float
    residual_rms_ms: float | None = None


@dataclass
class RegionGrid:
    times: np.ndarray
    inserted: np.ndarray           # bool: beat filled in where the tracker had no peak
    segments: list[Segment] = field(default_factory=list)
    step_changes: list[int] = field(default_factory=list)   # region-local beat j where a step starts


# --------------------------------------------------------------------- regions
def _robust_cv(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)) / med) if med > 0 else np.inf


def min_region_beats(duration: float) -> int:
    """Shortest run of beats that may form a grid (fewer allowed for short clips)."""
    return config.SHORT_CLIP_MIN_BEATS if duration < config.SHORT_CLIP_S else config.MIN_BEATS_PER_REGION


def find_regions(t: np.ndarray, h: np.ndarray, support: np.ndarray, duration: float = np.inf) -> list[np.ndarray]:
    """Split peak sequence into runs with a regular pulse.

    support: onset strength under each peak (same length as t).
    Returns list of index arrays into t.
    """
    n = len(t)
    min_beats = min_region_beats(duration)
    if n < min_beats:
        return []
    ibi = np.diff(t)
    local = median_filter(ibi, size=min(9, len(ibi) | 1), mode="nearest")
    breaks = np.flatnonzero(ibi > config.REGION_GAP_FACTOR * local)
    bounds = np.concatenate([[0], breaks + 1, [n]])
    regions = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = np.arange(a, b)
        idx = _trim_unsupported_edges(idx, support)
        if len(idx) < min_beats:
            continue
        if _robust_cv(np.diff(t[idx])) > config.REGION_MAX_IBI_CV:
            continue
        if h[idx].mean() < config.REGION_MIN_MEAN_ACTIVATION:
            continue
        regions.append(idx)
    return regions


def _trim_unsupported_edges(idx: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Drop leading/trailing beats with no onset under them.

    The tracker continues a pulse a few beats into silence or into a
    sustained pad; such edge beats have (near) zero onset strength. Interior
    beats are kept even if quiet -- a rest inside a groove is still on the grid.
    """
    if len(idx) == 0:
        return idx
    s = support[idx]
    ref = np.median(s)
    if ref <= 0:
        return idx[:0]
    ok = s >= config.EDGE_ONSET_MIN_REL * ref
    if not ok.any():
        return idx[:0]
    first, last = np.argmax(ok), len(ok) - 1 - np.argmax(ok[::-1])
    return idx[first:last + 1]


# ---------------------------------------------------------------------- repair
def _reference_period(ibi: np.ndarray) -> np.ndarray:
    """Per-interval reference period for repair decisions.

    A short running median follows tempo drift, but it also follows the
    tracker when it switches metrical level for a few bars (e.g. eighths <->
    dotted quarters in 6/8), which repair must undo, not preserve. So the
    reference is the running median, except where that disagrees with the
    region's dominant period by more than a plausible tempo drift; there the
    dominant period wins.
    """
    local = median_filter(ibi, size=min(9, len(ibi) | 1), mode="nearest")
    glob = np.median(ibi)
    off = np.abs(np.log(local / glob)) > np.log(1.0 / config.REPAIR_EXTRA_FACTOR)
    return np.where(off, glob, local)


def repair(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill single missed beats and drop doubled peaks, using the local period.

    Returns (times, inserted_mask). After repair consecutive entries are
    consecutive beats.
    """
    t = np.asarray(t, float)
    for _ in range(3):
        if len(t) < 3:
            break
        ibi = np.diff(t)
        local = _reference_period(ibi)
        short = np.flatnonzero(ibi / local < config.REPAIR_EXTRA_FACTOR)
        if not len(short):
            break
        # Interval i is too short: one of t[i], t[i+1] is an extra peak. Drop the
        # one whose removal leaves the gap around it closest to one local period.
        drop = set()
        for i in short:
            def err(c):
                if c <= 0 or c >= len(t) - 1:
                    return 0.5  # an endpoint: no gap to judge; neutral score
                return abs((t[c + 1] - t[c - 1]) / local[i] - 1.0)
            drop.add(i if err(i) < err(i + 1) else i + 1)
        t = np.delete(t, sorted(drop))
    inserted = np.zeros(len(t), bool)
    if len(t) >= 3:
        ibi = np.diff(t)
        local = _reference_period(ibi)
        out_t, out_ins = [t[0]], [False]
        for i, d in enumerate(ibi):
            m = int(round(d / local[i]))
            if d / local[i] >= config.REPAIR_MISSED_FACTOR and m >= 2:
                for q in range(1, m):
                    out_t.append(t[i] + d * q / m)
                    out_ins.append(True)
            out_t.append(t[i + 1])
            out_ins.append(False)
        t, inserted = np.array(out_t), np.array(out_ins)
    return t, inserted


# ------------------------------------------------------------------ fit tests
def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average; windows shrink at the ends (never pad with
    copies of the edge value, which would amplify a single edge residual)."""
    w = max(1, min(w, len(x)))
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    i = np.arange(n)
    a = np.clip(i - w // 2, 0, n)
    b = np.clip(i - w // 2 + w, 0, n)
    return (c[b] - c[a]) / (b - a)


def _poly_fit(j: np.ndarray, t: np.ndarray, deg: int):
    """Least squares with one round of outlier rejection (|r| > 25% of a beat)."""
    coef = np.polyfit(j, t, deg)
    r = t - np.polyval(coef, j)
    period = np.median(np.diff(t)) if len(t) > 1 else 1.0
    inl = np.abs(r) < config.FIT_OUTLIER_BEAT_FRACTION * period
    if inl.sum() >= max(deg + 2, int(0.5 * len(t))) and not inl.all():
        coef = np.polyfit(j[inl], t[inl], deg)
        r = t - np.polyval(coef, j)
    return coef, r, inl


def _passes(r: np.ndarray, inl: np.ndarray) -> bool:
    if inl.mean() < config.CONSTANT_MIN_INLIER_FRACTION:
        return False
    ri = r[inl]
    if np.sqrt(np.mean(ri ** 2)) > config.CONSTANT_MAX_RESIDUAL_RMS_S:
        return False
    sm = _moving_average(ri, config.CONSTANT_RESIDUAL_SMOOTH_BEATS)
    return bool(np.abs(sm).max() <= config.CONSTANT_MAX_SMOOTHED_RESIDUAL_S)


def fit_constant(t: np.ndarray):
    j = np.arange(len(t), dtype=float)
    coef, r, inl = _poly_fit(j, t, 1)
    return coef, r, inl, _passes(r, inl)


def fit_ramp(t: np.ndarray):
    j = np.arange(len(t), dtype=float)
    coef, r, inl = _poly_fit(j, t, 2)
    return coef, r, inl, _passes(r, inl)


# -------------------------------------------------------------- segmentation
class _LineSSE:
    """O(1) SSE of a least-squares line over any index range, via prefix sums."""

    def __init__(self, t: np.ndarray):
        j = np.arange(len(t), dtype=float)
        z = lambda a: np.concatenate([[0.0], np.cumsum(a)])
        self.S1, self.Sx, self.Sxx = z(np.ones_like(j)), z(j), z(j * j)
        self.Sy, self.Sxy, self.Syy = z(t), z(j * t), z(t * t)

    def __call__(self, a: int, b: int) -> tuple[float, float]:
        n = self.S1[b] - self.S1[a]
        sx, sxx = self.Sx[b] - self.Sx[a], self.Sxx[b] - self.Sxx[a]
        sy, sxy, syy = self.Sy[b] - self.Sy[a], self.Sxy[b] - self.Sxy[a], self.Syy[b] - self.Syy[a]
        vx = sxx - sx * sx / n
        cxy = sxy - sx * sy / n
        vy = syy - sy * sy / n
        slope = cxy / vx if vx > 0 else 0.0
        return max(0.0, vy - slope * cxy), slope


def _fit_guarded(t: np.ndarray, a: int, b: int, gl: int, gr: int):
    """Constant fit over [a, b) excluding `gl`/`gr` beats next to a step boundary.

    Returns (coef in piece-local index, residual over fitted part, inliers, passes).
    """
    lo, hi = a + gl, b - gr
    if hi - lo < config.MIN_BEATS_PER_REGION:
        lo, hi = a, b
    j = np.arange(lo - a, hi - a, dtype=float)
    coef, r, inl = _poly_fit(j, t[lo:hi], 1)
    return coef, r, inl, _passes(r, inl)


def _split_steps(t: np.ndarray, a: int, b: int, sse: _LineSSE, out: list,
                 gl: int = 0, gr: int = 0, depth: int = 0) -> bool:
    """Recursively split [a, b) into constant pieces. Returns False if impossible.

    gl/gr: beats at the left/right end that border a step change and are
    excluded from the constant test (the tracker smooths across a change, so
    the few beats on each side of it are pulled toward the other tempo).
    """
    if _fit_guarded(t, a, b, gl, gr)[3]:
        out.append((a, b, gl, gr))
        return True
    m = config.MIN_SEGMENT_BEATS
    if b - a < 2 * m or depth > 8:
        return False
    best, best_cost, slopes = None, np.inf, None
    for s in range(a + m, b - m + 1):
        c1, s1 = sse(a, s + 1)   # left piece includes the boundary beat
        c2, s2 = sse(s, b)
        if c1 + c2 < best_cost:
            best, best_cost, slopes = s, c1 + c2, (s1, s2)
    if best is None or min(slopes) <= 0:
        return False
    if abs(slopes[1] / slopes[0] - 1) < config.TEMPO_CHANGE_MIN_RATIO:
        return False
    g = config.STEP_TRANSITION_GUARD_BEATS
    return (_split_steps(t, a, best + 1, sse, out, gl, g, depth + 1)
            and _split_steps(t, best, b, sse, out, g, gr, depth + 1))


def _bpm(period: float) -> float:
    return float(60.0 / period) if period > 0 else 0.0


def fit_region(t: np.ndarray) -> RegionGrid:
    t, inserted = repair(t)
    n = len(t)
    j = np.arange(n, dtype=float)

    # Beats at region edges are guarded like beats at a step change: the
    # network lacks context there (it is warming up after silence or an
    # intro), and the first few beats come out early or late.
    G = config.STEP_TRANSITION_GUARD_BEATS
    coef, r, inl, ok = _fit_guarded(t, 0, n, G, G)
    if ok:
        g = np.polyval(coef, j)
        seg = Segment("constant", 0, n, g, _bpm(coef[0]), _bpm(coef[0]), float(1e3 * np.sqrt(np.mean(r[inl] ** 2))))
        return RegionGrid(g, inserted, [seg])

    pieces: list[tuple[int, int, int, int]] = []
    if _split_steps(t, 0, n, _LineSSE(t), pieces, G, G) and len(pieces) > 1:
        pieces.sort()
        g = np.empty(n)
        segs, steps = [], []
        for pi, (a, b, gl, gr) in enumerate(pieces):
            # pieces share their boundary beat; it belongs to the later piece
            c, rr, ii, _ = _fit_guarded(t, a, b, gl, gr)
            vals = np.polyval(c, np.arange(b - a, dtype=float))
            end = b - 1 if pi < len(pieces) - 1 else b
            g[a:end] = vals[: end - a]
            segs.append(Segment("constant", a, end, vals[: end - a], _bpm(c[0]), _bpm(c[0]),
                                float(1e3 * np.sqrt(np.mean(rr[ii] ** 2)))))
            if pi > 0:
                steps.append(a)
        return RegionGrid(g, inserted, segs, steps)

    coef, r, inl, ok = fit_ramp(t)
    if ok and n >= config.MIN_SEGMENT_BEATS:
        g = np.polyval(coef, j)
        d = np.polyval(np.polyder(coef), j)
        seg = Segment("ramp", 0, n, g, _bpm(d[0]), _bpm(d[-1]), float(1e3 * np.sqrt(np.mean(r[inl] ** 2))))
        return RegionGrid(g, inserted, [seg])

    g = smooth_free(t)
    ibi = np.diff(g)
    k = min(4, len(ibi))
    seg = Segment("free", 0, n, g, _bpm(np.median(ibi[:k])), _bpm(np.median(ibi[-k:])), None)
    return RegionGrid(g, inserted, [seg])


def smooth_free(t: np.ndarray) -> np.ndarray:
    """Savitzky-Golay (local quadratic) smoothing of tracked beat times.

    A local quadratic follows accelerando/ritardando and live-band drift, and
    averaging over the window removes most of the 20 ms frame quantisation.
    """
    from scipy.signal import savgol_filter
    w = 2 * config.VARIABLE_SMOOTH_HALF_WINDOW + 1
    if len(t) < w:
        return t.copy()
    return savgol_filter(t, w, 2, mode="interp")


# ---------------------------------------------------------------- merging
def merge_regions(rgrids: list[RegionGrid], support_fn) -> list[RegionGrid]:
    """Join neighbouring constant regions that are one unbroken grid.

    Two regions are merged when (1) both are single constant segments at the
    same tempo, (2) one straight line through both passes the constant test
    with an integer number of beats across the gap, and (3) the beats this
    would fill into the gap have onsets under them (median flux at least
    EDGE_ONSET_MIN_REL of the regions' median on-beat flux). (3) keeps a
    genuinely beatless breakdown from being filled with invented beats: a
    merge only happens where the tracker dropped out, not the music.
    support_fn(times) -> onset strength under each time.
    """
    if len(rgrids) < 2:
        return rgrids
    out = [rgrids[0]]
    for rg in rgrids[1:]:
        merged = _try_merge(out[-1], rg, support_fn)
        if merged is not None:
            out[-1] = merged
        else:
            out.append(rg)
    return out


def _try_merge(a: RegionGrid, b: RegionGrid, support_fn):
    if len(a.segments) != 1 or len(b.segments) != 1:
        return None
    sa, sb = a.segments[0], b.segments[0]
    if sa.kind != "constant" or sb.kind != "constant":
        return None
    if abs(sb.bpm_start / sa.bpm_start - 1) >= config.TEMPO_CHANGE_MIN_RATIO:
        return None
    period = 60.0 / sa.bpm_start
    gap_beats = (b.times[0] - a.times[-1]) / period
    k = int(round(gap_beats))
    if k < 2 or abs(gap_beats - k) > config.FIT_OUTLIER_BEAT_FRACTION:
        return None
    ja = np.arange(len(a.times), dtype=float)
    jb = np.arange(len(b.times), dtype=float) + len(a.times) - 1 + k
    jj, tt = np.concatenate([ja, jb]), np.concatenate([a.times, b.times])
    coef, r, inl = _poly_fit(jj, tt, 1)
    if not _passes(r, inl):
        return None
    fill = np.polyval(coef, np.arange(len(a.times), len(a.times) - 1 + k, dtype=float))
    ref = np.median(support_fn(tt))
    if ref <= 0 or np.median(support_fn(fill)) < config.EDGE_ONSET_MIN_REL * ref:
        return None
    n = len(a.times) + len(fill) + len(b.times)
    g = np.polyval(coef, np.arange(n, dtype=float))
    ins = np.concatenate([a.inserted, np.ones(len(fill), bool), b.inserted])
    seg = Segment("constant", 0, n, g, _bpm(coef[0]), _bpm(coef[0]), float(1e3 * np.sqrt(np.mean(r[inl] ** 2))))
    return RegionGrid(g, ins, [seg])
