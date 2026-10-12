"""Generalisation: random tempo / meter / offset tracks the code has never seen.

Seeds come from BEAT_AGENT_SYNTH_SEED (default fixed for reproducibility); set
it to anything to draw a new set. Beats are checked against generated truth at
the level the grid reports (compound meters may be tracked at the eighth note).
"""
import os

import mir_eval
import numpy as np
import pytest

from beat_agent.analysis.pipeline import analyze
from evals import synth

SEED = int(os.environ.get("BEAT_AGENT_SYNTH_SEED", "7"))
CASES = [synth.random_case(SEED * 100 + i, min_bpm=70, max_bpm=175) for i in range(6)]


def _truth_at_grid_level(truth, grid_bpm):
    ref = truth.beats
    truth_bpm = 60.0 / np.median(np.diff(ref))
    if truth.subdivision == 3 and grid_bpm and abs(grid_bpm / truth_bpm - 3) < 0.2:
        ref = np.sort(np.concatenate([ref[:-1] + np.diff(ref) * k / 3 for k in range(3)] + [ref[-1:]]))
    return ref


@pytest.mark.parametrize("kw", CASES, ids=[f"{c['meter']}@{c['segments'][0][1]}bpm+{c['start_s']}s" for c in CASES])
def test_random_track(tmp_path, kw):
    audio, truth = synth.make_track(**kw)
    g = analyze(synth.write(tmp_path / "x.wav", audio, 44100))
    est = g.beat_times()
    ref = _truth_at_grid_level(truth, g.tempo.bpm)
    f = mir_eval.beat.f_measure(ref, est, 0.07)
    assert f >= 0.9, (f, g.tempo.bpm, g.messages)
    assert g.tempo.type == "constant"
    assert est[0] >= truth.beat_start_s - 0.1, "beat emitted before the music starts"
    # grid invariants: indices contiguous, beat 0 is a downbeat in bar 1
    idx = [b.index for b in g.beats]
    assert idx == list(range(idx[0], idx[0] + len(idx)))
    b0 = g.beat(0)
    assert b0 is not None and b0.beat_in_bar == 1 and b0.bar == 1


def test_step_and_ramp(tmp_path):
    a, t = synth.make_track(segments=((48, 100, 100), (48, 112, 112)), seed=SEED)
    g = analyze(synth.write(tmp_path / "s.wav", a, 44100))
    assert g.tempo.type == "variable" and any(c["kind"] == "step" for c in g.tempo.changes)
    assert mir_eval.beat.f_measure(t.beats, g.beat_times(), 0.07) >= 0.95
    a, t = synth.make_track(segments=((80, 90, 115),), seed=SEED + 1)
    g = analyze(synth.write(tmp_path / "r.wav", a, 44100))
    assert g.tempo.type == "variable"
    assert mir_eval.beat.f_measure(t.beats, g.beat_times(), 0.07) >= 0.95
