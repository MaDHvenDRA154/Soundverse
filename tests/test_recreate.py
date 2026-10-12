"""Part 3: score schema, deterministic renderer, page safety, score ops."""
import json
import re

import numpy as np
import pytest

from beat_agent.render import score_ops as S
from beat_agent.render.html import render_html, safety_violations
from beat_agent.transcribe.pipeline import quantise, transcribe_drums
from beat_agent.transcribe.score import Note, Patch, Score, Track
from evals import synth


def _score():
    drums = Track(id="drums", name="Drums", role="drums", notes=[
        Note(start_beats=b, dur_beats=0.25, start_s=b * 0.5, dur_s=0.1, pitch=36 if b % 2 == 0 else 38, velocity=100,
             drum="kick" if b % 2 == 0 else "snare") for b in range(8)])
    bass = Track(id="bass", name="Bass", role="bass", patch=Patch(osc="saw", cutoff_hz=700),
                 notes=[Note(start_beats=b, dur_beats=1, start_s=b * 0.5, dur_s=0.5, pitch=40, velocity=90) for b in range(8)])
    return Score(tempo_bpm=120, analysed_tempo_bpm=120, time_signature=(4, 4), beats_per_bar=4, bars=2, loop_beats=8,
                 clip={"source": "x.wav", "start_s": 0.0, "end_s": 4.0, "bars": [1, 2], "mode": "whole_clip"},
                 tracks=[drums, bass], provenance={"llm": "none"})


def test_renderer_is_deterministic_and_safe():
    sc = _score()
    a, b = render_html(sc), render_html(sc)
    assert a == b
    assert safety_violations(a) == []
    payload = json.loads(re.search(r"const SCORE = (\{.*?\});\n", a, re.S).group(1))
    assert payload["tempo_bpm"] == 120 and len(payload["tracks"]) == 2


@pytest.mark.parametrize("bad", ['<script src="x.js">', "fetch('/x')", "import x from 'y'", '<audio src="a.wav">',
                                 "data:audio/wav;base64,AAAA", "https://example.com", "new XMLHttpRequest()"])
def test_safety_scan_catches(bad):
    assert safety_violations(f"<html>{bad}</html>")


def test_score_json_cannot_break_out_of_script():
    sc = _score()
    sc.clip["source"] = "</script><script>alert(1)</script>"
    html = render_html(sc)
    assert "</script><script>alert" not in html and safety_violations(html) == []


def test_score_ops():
    sc = _score()
    plan = S.ScorePlan.model_validate({"reading": "x", "ops": [{"op": "set_mute", "track": "drums", "mute": True},
                                                               {"op": "brightness", "track": "bass", "direction": "up"},
                                                               {"op": "set_tempo", "bpm": 140}]})
    new, status, echo = S.apply(sc, plan)
    assert status == "applied" and new.track("drums").mute and new.track("bass").patch.cutoff_hz == 1400
    assert new.tempo_bpm == 140 and new.track("bass").notes[3].start_beats == 3    # rhythm kept in beats
    assert "700 -> 1400 Hz" in echo and "120.00 -> 140.00 BPM" in echo
    assert not sc.track("drums").mute                                                 # original untouched
    bad = S.ScorePlan.model_validate({"reading": "x", "ops": [{"op": "set_mute", "track": "guitar", "mute": True}]})
    same, status, echo = S.apply(sc, bad)
    assert status == "error" and "no 'guitar' track" in echo and same is sc


def test_soft_quantise():
    assert quantise(np.array([1.1]), 0.25, 0.5)[0] == pytest.approx(1.05)
    assert quantise(np.array([1.1]), 0.25, 1.0)[0] == pytest.approx(1.0)


def test_drum_transcription_on_generated_hits():
    """Measured when written: kicks 12/12, hats 23/24 visible, snares 5/8 (snare body leaks into the kick band)."""
    sr = 44100
    rng = np.random.default_rng(0)
    k, s, h = synth._kick(sr, rng), synth._snare(sr, rng), synth._hat(sr, rng)
    x = np.zeros(int(sr * 9), np.float32)
    truth = []
    step = 60 / 110 / 4
    for i in range(64):
        t = 0.5 + i * step
        if i % 8 == 0 or i % 16 == 10:
            synth._add(x, k, int(t * sr), 0.9); truth.append((t, "kick"))
        if i % 8 == 4:
            synth._add(x, s, int(t * sr), 0.7); truth.append((t, "snare"))
        if i % 2 == 0:
            synth._add(x, h, int(t * sr), 0.8); truth.append((t, "hat"))
    est = transcribe_drums(x + 0.002 * rng.normal(size=len(x)).astype(np.float32), sr)

    def recall(kind):
        tt = [t for t, c in truth if c == kind]
        ee = np.array([t for t, c, _ in est if c == kind])
        return sum(len(ee) and np.min(np.abs(ee - t)) < 0.03 for t in tt) / len(tt)
    assert recall("kick") == 1.0
    assert recall("hat") >= 0.7
    assert recall("snare") >= 0.5
    onsets = np.array([t for t, _, _ in est])
    assert np.median([np.min(np.abs(onsets - t)) for t, _ in truth]) < 0.005
