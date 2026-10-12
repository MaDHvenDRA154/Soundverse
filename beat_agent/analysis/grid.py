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
        # regularity is judged after repair: doubled peaks in a fill (which
        # repair drops anyway) must not make a real pulse look irregular
        if _robust_cv(np.diff(repair(t[idx])[0])) > config.REGION_MAX_IBI_CV:
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
def _repair_period(ibi: np.ndarray) -> np.ndarray:
    """Per-interval baseline period for repair decisions.

    A short running median follows tempo drift, but it also follows the
    tracker when it switches metrical level for a few bars (e.g. eighths <->
    dotted quarters in 6/8), which repair must undo, not preserve. So the
    baseline is the running median, except where that disagrees with the
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
        local = _repair_period(ibi)
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
        local = _repair_period(ibi)
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


def _fit_guarded(t: np.ndarray, a: int, b: int, gl: int, gr: int, al: bool = False, ar: bool = False):
    """Constant fit over [a, b) excluding `gl`/`gr` beats next to a step boundary.

    Returns (coef in piece-local index, residual over fitted part, inliers, passes).
    """
    lo, hi = a + gl, b - gr
    if hi - lo < config.MIN_BEATS_PER_REGION:
        lo, hi = a, b
    j = np.arange(lo - a, hi - a, dtype=float)
    coef, r, inl = _poly_fit(j, t[lo:hi], 1)
    # Adaptive guard, only at STEP boundaries (al / ar): the tracker can stay
    # confused for more than the fixed guard around a change, and the step grid
    # rebuilds that stretch from both lines. At a region edge it must NOT grow:
    # beats there that do not fit mean the tempo changes there, which the step
    # model has to find (growing the guard extrapolated a 126 BPM line back over
    # beats that were still at 110 BPM).
    max_g = config.MIN_SEGMENT_BEATS // 2
    while al and gl and lo - a < max_g and hi - lo > config.MIN_BEATS_PER_REGION and abs(r[0]) > config.STEP_INLIER_S:
        lo += 1
        j = np.arange(lo - a, hi - a, dtype=float); coef, r, inl = _poly_fit(j, t[lo:hi], 1)
    while ar and gr and b - hi < max_g and hi - lo > config.MIN_BEATS_PER_REGION and abs(r[-1]) > config.STEP_INLIER_S:
        hi -= 1
        j = np.arange(lo - a, hi - a, dtype=float); coef, r, inl = _poly_fit(j, t[lo:hi], 1)
    return coef, r, inl, _passes(r, inl)


def _step_location(t: np.ndarray, a: int, b: int, m: int) -> int | None:
    """Most likely step-change beat in [a+m, b-m].

    For each candidate s, fit the old-tempo line to beats [a, s-g) and the
    new-tempo line to [s+g, b) (g = guard beats), then count the beats lying
    within STEP_INLIER_S of the line active at their position (old before s,
    new from s). Distances are to the *nearest* predicted beat, so a beat the
    tracker dropped or added while adjusting does not shift anything. Every
    beat assigned to the wrong side is an outlier, so the count peaks at the
    true change; ties go to the smaller residual. (Least squares over all beats
    was pulled up to 9 beats off by the tracker's transition errors, and a
    median-interval contrast up to 7; see notes/ai_mistakes_log.md.)
    """
    g = config.STEP_TRANSITION_GUARD_BEATS
    tol = config.STEP_INLIER_S

    def line(lo, hi):
        j = np.arange(lo, hi, dtype=float)
        c, _, _ = _poly_fit(j, t[lo:hi], 1)
        return c

    def dist(c, tt):
        k = np.round((tt - c[1]) / c[0])
        return np.abs(tt - (c[1] + c[0] * k))

    best, best_key = None, None
    for s in range(a + m, b - m + 1):
        lo1, hi1, lo2 = a, max(a + 4, s - g), min(b - 4, s + g)
        cl, cr = line(lo1, hi1), line(lo2, b)
        if cl[0] <= 0 or cr[0] <= 0:
            continue
        dl, dr = dist(cl, t[a:s]), dist(cr, t[s:b])
        inl = int((dl < tol).sum() + (dr < tol).sum())
        res = float(np.sum(np.minimum(dl, tol) ** 2) + np.sum(np.minimum(dr, tol) ** 2))
        key = (inl, -res)
        if best_key is None or key > best_key:
            best, best_key = s, key
    return best


def _split_steps(t: np.ndarray, a: int, b: int, sse: _LineSSE, out: list,
                 gl: int = 0, gr: int = 0, depth: int = 0, al: bool = False, ar: bool = False) -> bool:
    """Recursively split [a, b) into constant pieces. Returns False if impossible.

    gl/gr: beats at the left/right end that border a step change and are
    excluded from the constant test (the tracker smooths across a change, so
    the few beats on each side of it are pulled toward the other tempo).
    """
    if _fit_guarded(t, a, b, gl, gr, al, ar)[3]:
        out.append((a, b, gl, gr, al, ar))
        return True
    m = config.MIN_SEGMENT_BEATS
    if b - a < 2 * m or depth > 8:
        return False
    g = config.STEP_TRANSITION_GUARD_BEATS
    best = _step_location(t, a, b, m)
    if best is None:
        return False
    _, s1 = sse(a, max(a + 2, best + 1 - g))
    _, s2 = sse(min(b - 2, best + g), b)
    if min(s1, s2) <= 0 or abs(s2 / s1 - 1) < config.TEMPO_CHANGE_MIN_RATIO:
        return False
    return (_split_steps(t, a, best + 1, sse, out, gl, g, depth + 1, al, True)
            and _split_steps(t, best, b, sse, out, g, gr, depth + 1, True, ar))


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

    pieces: list[tuple] = []
    if _split_steps(t, 0, n, _LineSSE(t), pieces, G, G) and len(pieces) > 1:
        return _step_grid(t, sorted(pieces))

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


def _step_grid(t: np.ndarray, pieces: list[tuple[int, int, int, int]]) -> RegionGrid:
    """Grid for a piecewise-constant tempo, built from the fitted lines.

    At a step change both tempo lines pass through the change beat, so the
    change is placed where a beat of the old line and a beat of the new line
    coincide (searched within the guard zone). Beats are then generated from
    the lines -- old line before the crossing, new line from it -- which also
    corrects the extra / missing beats the tracker produces while it adjusts.
    """
    lines = []
    for a, b, gl, gr, al, ar in pieces:
        c, rr, ii, _ = _fit_guarded(t, a, b, gl, gr, al, ar)
        lines.append((a, b, c, float(1e3 * np.sqrt(np.mean(rr[ii] ** 2)))))
    G = config.STEP_TRANSITION_GUARD_BEATS + 2
    # crossing time between consecutive lines
    cuts = []
    for (a1, b1, c1, _), (a2, b2, c2, _) in zip(lines[:-1], lines[1:]):
        k1 = np.arange(b1 - a1 - 1 - G, b1 - a1 + G, dtype=float)       # around the shared boundary
        k2 = np.arange(-G, G + 1, dtype=float)
        t1, t2 = np.polyval(c1, k1), np.polyval(c2, k2)
        d = np.abs(t1[:, None] - t2[None, :])
        # Two tempo lines coincide again every LCM of their periods (7 beats at 140
        # ~= 6 beats at 120), so "closest pair" alone can pick a crossing several
        # beats past the change. A tempo change does not add or drop beats: prefer
        # the pair with the SAME global beat index on both lines, then +-1 (the
        # tracker dropped/added one), each only if the lines meet within a
        # quarter beat there.
        gap = (a1 + k1)[:, None] - (a2 + k2)[None, :]
        tol = config.FIT_OUTLIER_BEAT_FRACTION * min(abs(c1[0]), abs(c2[0]))
        i = j = None
        for shift in (0, 1, -1):
            m = np.where((gap == shift) & (d <= tol), d, np.inf)
            if np.isfinite(m).any():
                i, j = np.unravel_index(np.argmin(m), m.shape)
                break
        if i is None:
            i, j = np.unravel_index(np.argmin(d), d.shape)
        # the change beat is on both lines; it belongs to the new tempo, so the
        # old line stops just before its matching index k1[i]
        cuts.append((float(k1[i]), float(k2[j])))
    times, segs, steps = [], [], []
    for pi, (a, b, c, rms) in enumerate(lines):
        k_start = 0.0 if pi == 0 else cuts[pi - 1][1]
        k_stop = cuts[pi][0] if pi < len(lines) - 1 else float(b - a)   # exclusive
        ks = [float(np.polyval(c, k)) for k in np.arange(k_start, k_stop, 1.0)]
        j0 = len(times)
        if pi > 0:
            steps.append(j0)
        times.extend(ks)
        segs.append(Segment("constant", j0, len(times), np.array(ks), _bpm(c[0]), _bpm(c[0]), rms))
    times = np.array(times)
    # mark beats with no tracked peak within 35 ms as inserted
    nearest = np.min(np.abs(times[:, None] - t[None, :]), axis=1) if len(t) else np.full(len(times), np.inf)
    return RegionGrid(times, nearest > 0.035, segs, steps)


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


# ---------------------------------------------------------- phase refinement
def refine_phase(rg: RegionGrid, attack_fn) -> list[dict]:
    """Shift each tempo segment so its beats sit on the audio's attacks.

    The tracker's 20 ms frames leave each grid with a constant offset of a few
    to ~25 ms relative to the drum attacks (measured on generated audio). For
    every segment we measure attack - beat for each beat that has a percussive
    attack nearby, and shift the whole segment by the median, but only when
    that median is statistically clear (standard error <= PHASE_MAX_SE_S and
    |shift| >= 2 standard errors) and small (<= PHASE_MAX_SHIFT_S); otherwise
    the grid is left where the tracker put it. Tempo and spacing are untouched.
    attack_fn(times) -> attack time per beat (NaN where none).
    Returns a report per segment.
    """
    reports = []
    for s in rg.segments:
        seg_times = rg.times[s.j0:s.j1]
        att = attack_fn(seg_times)
        d = att - seg_times
        ok = np.isfinite(d)
        rep = {"start_beat_local": s.j0, "n_beats": int(len(seg_times)), "n_attacks": int(ok.sum()), "shift_ms": 0.0}
        if ok.sum() >= max(4, config.PHASE_MIN_SUPPORT_FRACTION * len(seg_times)):
            med = float(np.median(d[ok]))
            mad = float(np.median(np.abs(d[ok] - med)))
            se = 1.253 * 1.4826 * mad / np.sqrt(ok.sum())
            rep.update(median_offset_ms=round(1e3 * med, 2), se_ms=round(1e3 * se, 3))
            if se <= config.PHASE_MAX_SE_S and abs(med) >= 2 * se and abs(med) <= config.PHASE_MAX_SHIFT_S:
                rg.times[s.j0:s.j1] = seg_times + med
                s.times = s.times + med
                rep["shift_ms"] = round(1e3 * med, 2)
        reports.append(rep)
    return reports


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
    same = _try_merge_same_tempo(a, b, support_fn)
    return same if same is not None else _try_merge_joint(a, b, support_fn)


def _try_merge_joint(a: RegionGrid, b: RegionGrid, support_fn):
    """The tracker can drop out right around a tempo change, splitting it across
    two regions with the change a few beats into the second (too few beats to
    split there). If the gap continues `a`'s tempo line with onset-backed beats,
    refit the joined beats and keep the result only if the joint model is not
    'free' -- i.e. constant / step / ramp explain both sides together."""
    sa = a.segments[-1]
    if sa.kind != "constant":
        return None
    period = 60.0 / sa.bpm_end
    gap_beats = (b.times[0] - a.times[-1]) / period
    k = int(round(gap_beats))
    if k < 2 or abs(gap_beats - k) > config.FIT_OUTLIER_BEAT_FRACTION:
        return None
    fill = a.times[-1] + period * np.arange(1, k)
    ref = np.median(support_fn(np.concatenate([a.times, b.times])))
    if ref <= 0 or np.median(support_fn(fill)) < config.EDGE_ONSET_MIN_REL * ref:
        return None
    joined = fit_region(np.concatenate([a.times, fill, b.times]))
    if any(s.kind == "free" for s in joined.segments):
        return None
    if len(joined.segments) <= len(a.segments) + len(b.segments) - 1 or any(s.kind == "free" for s in b.segments):
        return joined
    return None


def _try_merge_same_tempo(a: RegionGrid, b: RegionGrid, support_fn):
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
