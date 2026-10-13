"""Executor, grid transform, validation and undo/redo -- no LLM involved.

Plans are built directly from the typed ops, exactly as the agent's validated
output would be.
"""
import mir_eval
import numpy as np
import pytest
import soundfile as sf

from beat_agent.analysis.pipeline import analyze
from beat_agent.edit import ops as O
from beat_agent.edit.session import EditSession
from beat_agent.edit.timeline import Clip, Timeline, render
from beat_agent.io.audio import open_source
from evals import synth
from evals.clicks import splice_scores


def plan(*ops, reading="test"):
    return O.EditPlan.model_validate({"reading": reading, "ops": list(ops)})


def bars(s, e):
    return {"unit": "bars", "start": s, "end": e}


def beats(s, e):
    return {"unit": "beats", "start": s, "end": e}


# ------------------------------------------------------------------ timeline
def test_timeline_ops_lengths_and_sources():
    t = Timeline.whole(1000)
    assert t.remove(100, 200).length == 900
    r = t.repeat(100, 200, 3)
    assert r.length == 1200
    # the last copy is contiguous with the rest of the track, so the two clips merge
    assert [(c.src_start, c.src_end) for c in r.clips] == [(0, 200), (100, 200), (100, 1000)]
    m = t.move(100, 200, 500)
    assert m.length == 1000
    assert [(c.src_start, c.src_end) for c in m.clips] == [(0, 100), (200, 500), (100, 200), (500, 1000)]
    assert t.keep_only(300, 400).clips == (Clip(300, 400),)
    assert t.insert_copy(0, 50, 1000, 2).length == 1100
    assert r.splices() == [200, 300]


@pytest.fixture(scope="module")
def ramp_src(tmp_path_factory):
    d = tmp_path_factory.mktemp("ramp")
    rng = np.random.default_rng(0)
    x = rng.integers(-20000, 20000, (44100, 2)).astype(np.int16)
    sf.write(d / "x.wav", x, 44100, subtype="PCM_16")
    return open_source(d / "x.wav"), x, d


def test_render_is_sample_accurate_and_bit_exact(ramp_src):
    src, x, d = ramp_src
    tl = Timeline.whole(src.frames).remove(10000, 20000)
    r = render(src, tl, d / "o.wav")
    y, _ = sf.read(d / "o.wav", dtype="int16")
    assert len(y) == 44100 - 10000 and r["splices"] == [10000]
    H = int(round(0.005 * 44100 / 2))
    # away from the splice the output is the source, bit for bit
    assert np.array_equal(y[:10000 - H], x[:10000 - H])
    assert np.array_equal(y[10000 + H:], x[20000 + H:])
    assert sf.info(d / "o.wav").subtype == "PCM_16"


@pytest.mark.parametrize("sr,subtype,ch", [(8000, "PCM_16", 1), (48000, "PCM_24", 2), (96000, "FLOAT", 2)])
def test_render_keeps_rate_depth_channels(tmp_path, sr, subtype, ch):
    sf.write(tmp_path / "x.wav", 0.1 * np.ones((sr, ch), np.float32), sr, subtype=subtype)
    src = open_source(tmp_path / "x.wav")
    render(src, Timeline.whole(src.frames).repeat(0, sr // 4, 2), tmp_path / "o.wav")
    i = sf.info(tmp_path / "o.wav")
    assert (i.samplerate, i.subtype, i.channels, i.frames) == (sr, subtype, ch, sr + sr // 4)


def test_crossfade_removes_click_and_detector_sees_hard_cut(tmp_path):
    sr = 44100
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.02 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    sf.write(tmp_path / "s.wav", x, sr, subtype="FLOAT")
    src = open_source(tmp_path / "s.wav")
    # splice two points at very different phases of the 220 Hz wave
    tl = Timeline.whole(src.frames).remove(10000, 10000 + 51)
    render(src, tl, tmp_path / "soft.wav")
    render(src, tl, tmp_path / "hard.wav", crossfade_s=2 / sr)   # ~no crossfade: negative control
    sp = tl.splice_points()
    soft = splice_scores(sf.read(tmp_path / "soft.wav")[0], x, sr, sp)[0]
    hard = splice_scores(sf.read(tmp_path / "hard.wav")[0], x, sr, sp)[0]
    assert hard["flagged"] and hard["step_ratio"] > 10, hard
    assert not soft["flagged"], soft


def test_fade_and_gain_envelopes(tmp_path):
    sr = 8000
    sf.write(tmp_path / "x.wav", np.ones((sr, 1), np.float32) * 0.5, sr, subtype="FLOAT")
    src = open_source(tmp_path / "x.wav")
    from beat_agent.edit.timeline import Factor
    tl = Timeline.whole(sr).apply_factor(4000, 8000, Factor("fade_out")).apply_factor(0, 2000, Factor("gain", value=0.5))
    render(src, tl, tmp_path / "o.wav")
    y = sf.read(tmp_path / "o.wav")[0]
    assert np.allclose(y[:2000], 0.25) and np.allclose(y[2000:4000], 0.5)
    assert abs(y[6000] - 0.5 * np.cos(np.pi / 4)) < 1e-3 and abs(y[-1]) < 1e-3


# ------------------------------------------------------------- session
@pytest.fixture(scope="module")
def track(tmp_path_factory):
    d = tmp_path_factory.mktemp("trk")
    a, truth = synth.make_track(segments=((96, 120, 120),), start_s=0.5, seed=3, four_on_floor=True)
    p = synth.write(d / "t.wav", a, 44100, "PCM_24")
    src = open_source(p)
    return src, analyze(p, src), truth, d


def test_cut_and_repeat_keeps_grid_true(track):
    src, g, truth, d = track
    s = EditSession(src, g)
    r = s.apply(plan({"op": "repeat", "range": beats(16, 32), "times_total": 2}))
    assert r.status == "applied", r.message
    period = 0.5
    assert abs(r.length_after_s - r.length_before_s - 16 * period) < 0.01
    eg = s.grid()
    assert len(eg.beats) == len(g.beats) + 16
    assert all(b.complete for b in eg.bars if 1 <= b.bar < eg.n_bars())   # bar-aligned edit: no partial bars
    out = s.export(d, "rep")
    assert sf.info(out["audio"]).subtype == "PCM_24"
    # re-detect on the edited audio and compare with the transformed grid
    re = analyze(out["audio"])
    f = mir_eval.beat.f_measure(eg.beat_times(), re.beat_times(), 0.07)
    assert f >= 0.97, f
    dt = [np.min(np.abs(re.beat_times() - t)) for t in eg.beat_times()]
    assert np.median(dt) < 0.010
    fd = mir_eval.beat.f_measure(np.array(eg.downbeats), np.array(re.downbeats), 0.07)
    assert fd >= 0.9, fd
    sc = splice_scores(sf.read(out["audio"])[0], sf.read(src.path)[0], src.samplerate, out["splice_points"])
    assert sc and not any(x["flagged"] for x in sc), sc


def test_remove_bar_resolves_against_current_grid(track):
    src, g, _, _ = track
    s = EditSession(src, g)
    s.apply(plan({"op": "remove", "range": bars(1, 1)}))
    g1 = s.grid()
    # bar 5 now is what was bar 6
    assert abs(g1.bar(5).end - g1.bar(5).start - 2.0) < 0.01
    before = g1.bar(5)
    r = s.apply(plan({"op": "remove", "range": bars(5, 5)}))
    assert r.status == "applied"
    assert abs(r.resolved[0]["seconds"][0] - before.start) < 1e-4


def test_out_of_range_is_an_error_with_real_limits(track):
    src, g, _, _ = track
    s = EditSession(src, g)
    r = s.apply(plan({"op": "remove", "range": bars(200, 210)}))
    assert r.status == "error"
    assert f"bars 1-{g.n_bars()}" in r.message and "Nothing was changed" in r.message
    assert s.length_s() == pytest.approx(src.duration)
    assert s.apply(plan({"op": "remove", "range": beats(8, 8)})).status == "error"   # empty


def test_turn_is_atomic(track):
    src, g, _, _ = track
    s = EditSession(src, g)
    r = s.apply(plan({"op": "remove", "range": bars(2, 2)}, {"op": "remove", "range": bars(999, 999)}))
    assert r.status == "error"
    assert s.length_s() == pytest.approx(src.duration) and not s.undo_stack


def test_undo_redo_per_turn(track):
    src, g, _, _ = track
    s = EditSession(src, g)
    L0 = s.length_s()
    s.apply(plan({"op": "remove", "range": bars(2, 3)}, {"op": "gain", "range": {"unit": "whole_track"}, "db": -3}))
    L1 = s.length_s()
    assert s.apply(plan({"op": "undo"})).status == "undo" and s.length_s() == pytest.approx(L0)
    assert s.apply(plan({"op": "redo"})).status == "redo" and s.length_s() == pytest.approx(L1)
    assert s.apply(plan({"op": "redo"})).status == "error"


def test_fade_last_beats_reaches_the_end(track):
    src, g, _, d = track
    s = EditSession(src, g)
    r = s.apply(plan({"op": "fade", "range": {"unit": "last_beats", "count": 8}, "direction": "out"}))
    assert r.status == "applied" and r.resolved[0]["seconds"][1] == pytest.approx(src.duration)
    y = sf.read(s.export(d, "fade")["audio"])[0]
    assert np.abs(y[-200:]).max() < 1e-3


def test_no_tempo_change_means_no_guess(track):
    src, g, _, _ = track
    s = EditSession(src, g)
    r = s.apply(plan({"op": "repeat", "range": {"unit": "bars_before_tempo_change"}, "times_total": 2}))
    assert r.status == "error" and "No tempo change was detected" in r.message


def test_variable_tempo_bar_before_change(tmp_path):
    a, truth = synth.make_track(segments=((64, 110, 110), (64, 126, 126)), start_s=0.5, seed=4)
    p = synth.write(tmp_path / "v.wav", a, 44100)
    src = open_source(p)
    g = analyze(p, src)
    s = EditSession(src, g)
    r = s.apply(plan({"op": "repeat", "range": {"unit": "bars_before_tempo_change"}, "times_total": 2}))
    assert r.status == "applied", r.message
    a0, a1 = r.resolved[0]["seconds"]
    change_t = truth.beats[64]
    assert abs(a1 - change_t) < 0.01                    # ends on the change
    assert abs((a1 - a0) - 4 * 60 / 110) < 0.01         # one bar at the OLD tempo, not 60/bpm_global


def test_proposal_does_not_cut_and_confirm_does(track):
    src, g, _, _ = track
    s = EditSession(src, g)
    r = s.apply(plan({"op": "propose_cut"}))
    assert r.status == "proposal" and "not cut anything" in r.message
    assert s.length_s() == pytest.approx(src.duration)
    b = r.resolved[0]["bars"]
    r2 = s.apply(plan({"op": "confirm_proposal"}))
    assert r2.status == "applied"
    assert s.length_s() < src.duration - (b["end"] - b["start"] + 1) * 2.0 + 0.01


def test_grid_corrections(track):
    src, g, _, _ = track
    s = EditSession(src, g)
    n = len(g.beats)
    bars_before = [b.start for b in g.bars]
    s.apply(plan({"op": "set_tempo_octave", "factor": 2.0}))
    g2 = s.grid()
    assert len(g2.beats) == 2 * n - 1 and g2.tempo.bpm == pytest.approx(240, rel=0.01)
    assert g2.time_signature.beats_per_bar == 8 and [b.start for b in g2.bars] == pytest.approx(bars_before, abs=1e-4)
    s.apply(plan({"op": "set_tempo_octave", "factor": 0.5}))
    g3 = s.grid()
    assert len(g3.beats) == n and [b.start for b in g3.bars] == pytest.approx(bars_before, abs=1e-4)
    s.apply(plan({"op": "undo"}))
    s.apply(plan({"op": "undo"}))
    s.apply(plan({"op": "set_beats_per_bar", "beats_per_bar": 3}))
    assert s.grid().time_signature.numerator == 3
    s.apply(plan({"op": "shift_downbeat", "beats": 1}))
    assert s.grid().beats[0].beat_in_bar != 1 or s.grid().beats[0].index < 0


def test_undo_names_the_turn(track):
    src, g, _, _ = track
    s = EditSession(src, g)
    s.apply(plan({"op": "remove", "range": bars(2, 2)}), "take out bar 2")
    r = s.apply(plan({"op": "undo"}), "undo")
    assert "'take out bar 2'" in r.message
    assert "'take out bar 2'" in s.apply(plan({"op": "redo"}), "redo").message


# ------------------------------------------------------------------ repeat length (regression, live-app bug)
@pytest.fixture(scope="module")
def ramp68(tmp_path_factory):
    """6/8 with a tempo ramp: a variable grid, like the track the bug was found on."""
    sr = 44100
    audio, _ = synth.make_track(sr=sr, segments=((48, 66, 80),), meter="6/8", seed=7)
    p = tmp_path_factory.mktemp("r68") / "ramp68.wav"
    synth.write(p, audio, sr)
    src = open_source(str(p))
    return src, analyze(str(p), src)


def _span(sess, r):
    g = sess.grid()
    t = {b.index: b.time for b in g.beats}
    return int(round(t[r["end"]] * sess.sr)) - int(round(t[r["start"]] * sess.sr))


@pytest.mark.parametrize("times_total", [2, 3])
def test_repeat_adds_range_length_on_variable_68_grid(ramp68, times_total):
    src, grid = ramp68
    assert grid.tempo.type != "constant" and grid.time_signature.numerator in (6, 12)
    sess = EditSession(src, grid)
    n0, span = sess.state.timeline.length, _span(sess, beats(16, 32))
    res = sess.apply(plan({"op": "repeat", "range": beats(16, 32), "times_total": times_total}))
    assert res.status == "applied"
    # centred crossfades use source audio beyond the clip edges, so they overlap nothing: no length lost
    assert abs(sess.state.timeline.length - (n0 + (times_total - 1) * span)) <= 2


def test_cut_and_repeat_plan_with_a_remove_is_read_as_a_selection(ramp68):
    """The live app got [remove 16-32, repeat 16-32 x2] from the model for 'Cut beat 16-32 and repeat it
    twice': the remove deleted the range and the repeat then doubled the next 16 beats (net -8 ms)."""
    src, grid = ramp68
    sess = EditSession(src, grid)
    n0, span = sess.state.timeline.length, _span(sess, beats(16, 32))
    res = sess.apply(plan({"op": "remove", "range": beats(16, 32)},
                          {"op": "repeat", "range": beats(16, 32), "times_total": 2}))
    assert res.status == "applied" and "not deleting it" in res.message
    assert abs(sess.state.timeline.length - (n0 + span)) <= 2
    # a remove of a *different* range before a repeat is left alone
    sess2 = EditSession(src, grid)
    sess2.apply(plan({"op": "remove", "range": beats(0, 8)}, {"op": "repeat", "range": beats(16, 32), "times_total": 2}))
    assert sess2.state.timeline.length < n0 + span
