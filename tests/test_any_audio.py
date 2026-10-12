"""Robustness: any input either yields a correct grid or a clear message.

Runs the generated-audio suite (`evals/robustness.py`) through the full public
pipeline. Known limitations are *reported* (status "partial"/"fail" in
evals/results/robustness.md), not hidden: this test fails on crashes, on
fabricated beats for beatless input, on format changes, and if the share of
correct-or-usable grids drops below the floor.
"""
import pytest

from evals import robustness

# Measured when written: 31 of 33 non-slow cases pass or partial (the two
# failures are fast 3/4 read as 4/4 and as 6/8). Dropping below this floor is a
# regression.
MIN_OK_SHARE = 0.85


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("rob")
    return {r.name: r for r in robustness.run(quick=False, include_slow=False, out_dir=out,
                                              work_dir=out / "work")}


def test_no_crashes(results):
    crashed = [n for n, r in results.items() if any("crash" in x for x in r.notes)]
    assert not crashed, crashed


def test_beatless_inputs_get_a_message_not_a_grid(results):
    for name in ("silence_20s", "white_noise_20s"):
        r = results[name]
        assert r.status == "pass", (name, r.notes)
        assert r.checks["n_beats"] == 0 and r.checks["message"]


def test_no_beats_in_drumless_intro(results):
    r = results["pad_intro_45s"]
    assert r.checks["beats_before_music"] == 0
    assert r.status == "pass", r.notes


@pytest.mark.parametrize("name", ["sr_8k_mono", "sr_96k_stereo_24bit", "float32_6ch_48k", "flac_16bit",
                                  "mp3_roundtrip", "m4a_aac_roundtrip", "ogg_vorbis", "short_3s"])
def test_formats_and_lengths(results, name):
    r = results[name]
    assert r.status == "pass", (name, r.checks, r.notes)


def test_tempo_changes_detected(results):
    for name in ("tempo_step_118_to_126", "tempo_step_140_to_120"):
        r = results[name]
        assert r.checks.get("tempo_change_found"), r.notes
        assert r.status == "pass", (name, r.notes)
    assert results["tempo_ramp_96_to_120"].checks["tempo_type"] == "variable"


def test_share_of_usable_grids(results):
    music = [r for r in results.values() if "message" not in r.checks]
    ok = [r for r in music if r.status in ("pass", "partial")]
    share = len(ok) / len(music)
    assert share >= MIN_OK_SHARE, [(r.name, r.notes) for r in music if r.status == "fail"]


@pytest.mark.slow
def test_twelve_minute_clip(tmp_path):
    rs = robustness.run(include_slow=True, out_dir=tmp_path, work_dir=tmp_path / "w", only=["long_12min"])
    r = rs[0]
    assert r.status == "pass", r.notes
    assert r.runtime_s < 120, r.runtime_s
