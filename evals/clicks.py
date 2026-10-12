"""Splice click detector (eval-only).

A splice is judged against the audio that *should* be there. For a splice at
output sample p joining source A (ending at a_end) to source B (starting at
b_start), we measure a +-3 ms zone in three places:
  out  : the edited audio around p
  refA : the source around a_end (what A would have played next)
  refB : the source around b_start (how B itself begins)
and report, for two measures,
  jump  = max |x[n] - x[n-1]|                  (step discontinuity)
  hf    = mean energy above min(8 kHz, 0.4 sr)  (broadband click energy)
ratio = out / max(refA, refB). A clean splice adds nothing that is not
already in the source, so the ratio is ~1 or below; a kick that happens to
start on the splice is in refB too and does not count.

Primary measure, step_ratio: the sample-to-sample step exactly at the splice
divided by the 99th percentile of natural steps within +-1 ms of the two
source points. A drum attack ramps over several samples; a hard cut between
disagreeing waveforms jumps in one. Zone-based ratios could not tell a hard
cut from a crossfade on real music (the drum attack dominates the zone), so
they are reported but only the step ratio and audible HF decide the flag.
A hard cut is itself only a click where the two waveforms disagree; the
negative control in the report shows where that was the case.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

ZONE_S = 0.003
# Flag when either ratio exceeds 2: the splice zone has twice the step size
# or HF energy of the worse of the two source zones it was built from.
FLAG_RATIO = 2.0
# Flag when the step at the splice exceeds the largest 1% of natural steps
# around the two source points (step_ratio > 1).
STEP_FLAG = 1.0
STEP_CONTEXT_S = 0.001
# ...but an HF excess only counts if it is audible: HF energy at least -50 dB
# relative to the total signal energy in the same zone (below that it is
# masked). Without this, a correct crossfade on material with no HF at all
# (a pure low tone) scores a huge ratio of two near-zero numbers.
HF_AUDIBLE_REL = 1e-5


def _measures(x: np.ndarray, hf: np.ndarray, p: int, z: int) -> tuple[float, float]:
    lo, hi = max(1, p - z), min(len(x), p + z)
    if hi - lo < 2:
        return 0.0, 0.0
    return float(np.abs(np.diff(x[lo - 1:hi], axis=0)).max()), float((hf[lo:hi] ** 2).mean())


def _hp(x: np.ndarray, sr: int) -> np.ndarray:
    sos = butter(4, min(8000.0, 0.4 * sr), btype="high", fs=sr, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def splice_scores(out: np.ndarray, src: np.ndarray, sr: int, splices: list[tuple[int, int, int]]) -> list[dict]:
    """out/src: (n,) or (n, ch) float arrays at the same rate. splices: Timeline.splice_points()."""
    out = out[:, None] if out.ndim == 1 else out
    src = src[:, None] if src.ndim == 1 else src
    ho, hs = _hp(out, sr), _hp(src, sr)
    z = max(2, int(ZONE_S * sr))
    res = []
    for p, a_end, b_start in splices:
        jo, eo = _measures(out, ho, p, z)
        ja, ea = _measures(src, hs, a_end, z)
        jb, eb = _measures(src, hs, b_start, z)
        jr = jo / max(ja, jb, 1e-6)
        hr = eo / max(ea, eb, 1e-12)
        lo, hi = max(0, p - z), min(len(out), p + z)
        hf_rel = eo / max(float((out[lo:hi] ** 2).mean()), 1e-12)
        w = max(2, int(STEP_CONTEXT_S * sr))
        step = float(np.abs(out[p] - out[p - 1]).max()) if 0 < p < len(out) else 0.0
        nat = np.concatenate([np.abs(np.diff(src[max(0, q - w):q + w], axis=0)).ravel() for q in (a_end, b_start)])
        sr_ = step / (float(np.percentile(nat, 99)) + 1e-9) if len(nat) else float("nan")
        res.append({"splice_s": round(p / sr, 4), "step_ratio": round(sr_, 3), "jump_ratio": round(jr, 3),
                    "hf_ratio": round(hr, 3), "hf_rel_db": round(10 * np.log10(hf_rel + 1e-20), 1),
                    "flagged": bool(sr_ > STEP_FLAG or (hr > FLAG_RATIO and hf_rel >= HF_AUDIBLE_REL))})
    return res
