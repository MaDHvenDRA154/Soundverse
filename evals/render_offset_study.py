"""Where does the render-vs-original onset offset on the Part 3 drum clips come from?

The Part 3 eval reports the recreated page's onsets landing a median 17.4 ms
*after* the original's on subP1 (11.6 ms on subP2/subP4). Candidates:
  scheduling   the page plays notes late (look-ahead scheduler, compressor delay)
  transcription/quantise  the score's note times are already late of the audio
  attack shape the onset detector fires later on our synth voices than on the
               soundfont samples, though both start at the same instant

Every term is measured, in ms, positive = later, on drum notes only:
  T0  the grid's beats - the nearest reference MIDI drum onset (what the
      quantiser pulls notes towards)
  T1  raw drum detections (before quantise) - reference MIDI onsets
  T2  score note times (after 0.5 soft quantise) - raw detections
  T3  page signal path: first audible sample - scheduled time, with the
      shipped compressor and with it bypassed (diagnostic copy only)
  T4  detector on our synth voices: librosa onsets on an isolated-hit probe
      render - scheduled times
  T5  detector on the original: librosa onsets on the substitute audio -
      reference MIDI onsets
  T6  the reported number recomputed: render onsets - original onsets
  predicted T6 ~= (T1 + T2) + T4 - T5
librosa onsets are measured with the eval's own settings (hop 128 at 22.05 kHz,
5.8 ms frames), and again at hop 32 to see how much is frame quantisation.
Changes nothing; numbers go to evals/results/render_offset_study.{md,json}.
    python -m evals.render_offset_study
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

CLIPS = ("subP1", "subP2")
OUT = Path("evals/out/part3")
WIN = 0.05


def _signed(est, ref, win=WIN):
    est = np.asarray(est, float)
    d = [est[np.argmin(np.abs(est - r))] - r for r in ref if len(est)]
    d = np.array([x for x in d if abs(x) <= win])
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "median_ms": round(float(1e3 * np.median(d)), 2),
            "mean_ms": round(float(1e3 * d.mean()), 2), "p10_ms": round(float(1e3 * np.percentile(d, 10)), 2),
            "p90_ms": round(float(1e3 * np.percentile(d, 90)), 2)}


def _librosa_onsets(path, hop):
    import librosa
    y, _ = librosa.load(str(path), sr=22050, mono=True)
    return librosa.onset.onset_detect(y=y, sr=22050, units="time", backtrack=False, hop_length=hop)


def _merge(times, tol=0.010):
    out = []
    for t in sorted(times):
        if not out or t - out[-1] > tol:
            out.append(t)
    return np.array(out)


def _first_audible(y, sr, when, thresh=1e-3, look=0.03):
    a = int(round(when * sr))
    seg = np.abs(y[a:a + int(look * sr)]).max(axis=1) if y.ndim == 2 else np.abs(y[a:a + int(look * sr)])
    k = np.flatnonzero(seg > thresh)
    return (a + k[0]) / sr - when if len(k) else None


def _probe(score, bypass_comp: bool, td: Path):
    """Isolated hits at known times: kick, snare, hat, one every 2 beats, 4 rounds."""
    from beat_agent.render.html import render_html
    from beat_agent.render.offline import render_offline
    from beat_agent.transcribe.score import Note, Score, Track
    spb = 60.0 / score.tempo_bpm
    kinds = [("kick", 36), ("snare", 38), ("hat", 42)] * 4
    notes = [Note(start_beats=2.0 * i, dur_beats=0.25, start_s=2.0 * i * spb, dur_s=0.25 * spb, pitch=p,
                  velocity=100, drum=k) for i, (k, p) in enumerate(kinds)]
    lb = 2.0 * len(kinds)
    sc = Score(tempo_bpm=score.tempo_bpm, analysed_tempo_bpm=score.tempo_bpm, time_signature=(4, 4),
               beats_per_bar=4, bars=min(16, int(np.ceil(lb / 4))), loop_beats=lb,
               clip={"note": "diagnostic probe"}, tracks=[Track(id="drums", name="Drums", role="drums", notes=notes)])
    html = render_html(sc)
    if bypass_comp:
        assert "master.connect(comp)" in html
        html = html.replace("master.connect(comp)", "master.connect(out)")
    page = td / f"probe_{int(bypass_comp)}.html"
    page.write_text(html, encoding="utf-8")
    wav = td / f"probe_{int(bypass_comp)}.wav"
    render_offline(page, wav, loops=1)
    return wav, np.array([n.start_s for n in notes]), [k for k, _ in kinds]


def study(cid: str) -> dict:
    import pretty_midi
    from beat_agent import config
    from beat_agent.analysis.pipeline import analyze
    from beat_agent.io.audio import open_source
    from beat_agent.transcribe.pipeline import plan_loop, transcribe_drums
    from beat_agent.transcribe.score import Score
    meta = json.loads(Path("evals/substitutes/part3/clips.json").read_text())[cid]
    d = OUT / cid
    score = Score.model_validate_json((d / "score.json").read_text())
    ref = _merge(n.start for i in pretty_midi.PrettyMIDI(meta["midi"]).instruments if i.is_drum for n in i.notes)
    src = open_source(meta["audio"])
    grid = analyze(meta["audio"], src)
    plan = plan_loop(grid, src, None, None)
    beats = np.array([b.time for b in grid.beats])
    t0_d = np.array([b - ref[np.argmin(np.abs(ref - b))] for b in beats])
    t0_d = t0_d[np.abs(t0_d) <= WIN]
    stem, ssr = sf.read(d / "stems" / "drums.wav")
    off = plan["t0"] - config.DEMUCS_CONTEXT_S
    raw = np.array([t + off for t, _, _ in transcribe_drums(stem, ssr)])
    raw = raw[(raw >= plan["t0"] - 0.05) & (raw < plan["t1"])]
    drum_track = score.track("drums")
    sched = np.array([n.start_s for n in drum_track.notes])
    sched_render = np.array([n.start_beats * 60.0 / score.tempo_bpm for n in drum_track.notes])
    res = {"clip": cid, "source": meta["source"], "tempo_bpm": score.tempo_bpm,
           "T0_grid_beats_minus_reference": {"n": int(len(t0_d)), "median_ms": round(float(1e3 * np.median(t0_d)), 2),
                                             "p10_ms": round(float(1e3 * np.percentile(t0_d, 10)), 2),
                                             "p90_ms": round(float(1e3 * np.percentile(t0_d, 90)), 2)} if len(t0_d) else {"n": 0},
           "quantize_strength": config.QUANTIZE_STRENGTH,
           "T1_raw_detection_minus_reference": _signed(raw, ref),
           "T2_score_minus_raw_detection": _signed(sched, raw),
           "T1+T2_score_minus_reference": _signed(sched, ref)}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for bypass in (False, True):
            wav, when, kinds = _probe(score, bypass, td)
            y, sr = sf.read(wav, always_2d=True)
            path = {k: [] for k in ("kick", "snare", "hat")}
            for w, k in zip(when, kinds):
                v = _first_audible(y, sr, w)
                if v is not None:
                    path[k].append(1e3 * v)
            res[f"T3_signal_path_{'compressor_bypassed' if bypass else 'shipped'}_ms"] = {
                k: round(float(np.median(v)), 3) for k, v in path.items() if v}
            if not bypass:
                for hop in (128, 32):
                    o = _librosa_onsets(wav, hop)
                    res[f"T4_detector_on_synth_hop{hop}"] = {
                        k: _signed(o, when[[i for i, kk in enumerate(kinds) if kk == k]]) for k in ("kick", "snare", "hat")}
                    res[f"T4_detector_on_synth_hop{hop}"]["all"] = _signed(o, when)
    for hop in (128, 32):
        o0 = _librosa_onsets(meta["audio"], hop)
        o1 = _librosa_onsets(d / "render_1loop.wav", hop) + plan["t0"]
        res[f"T5_detector_on_original_hop{hop}"] = _signed(o0, ref)
        res[f"T6_render_minus_original_hop{hop}"] = _signed(o1, o0)
        res[f"render_onsets_minus_scheduled_hop{hop}"] = _signed(o1 - plan["t0"], sched_render)
    g = lambda k: (res[k] or {}).get("median_ms")
    for hop in (128, 32):
        try:
            res[f"predicted_T6_hop{hop}_ms"] = round(g("T1+T2_score_minus_reference")
                                                     + res[f"T4_detector_on_synth_hop{hop}"]["all"]["median_ms"]
                                                     - g(f"T5_detector_on_original_hop{hop}"), 2)
        except (TypeError, KeyError):
            res[f"predicted_T6_hop{hop}_ms"] = None
    return res


def run(out="evals/results"):
    rows = [study(c) for c in CLIPS if (OUT / c / "score.json").exists()]
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "render_offset_study.json").write_text(json.dumps(rows, indent=1))
    (Path(out) / "render_offset_study.md").write_text(report(rows))
    print(report(rows))


def _c(s):
    return "-" if not s or not s.get("n") else f"{s['median_ms']:+.1f} (n={s['n']}, p10 {s['p10_ms']:+.1f}, p90 {s['p90_ms']:+.1f})"


def report(rows) -> str:
    L = ["# Render offset study (Part 3, SUBSTITUTE clips)", "",
         "Generated by `python -m evals.render_offset_study` after `python -m evals.run part3`. Drum notes only;",
         "ms, positive = later; median (n, p10, p90). See the module docstring for the method.", ""]
    for r in rows:
        L += [f"## {r['clip']} -- {r['source']} ({r['tempo_bpm']} BPM)", "",
              "| term | what | hop 128 (eval setting) | hop 32 |", "|---|---|---|---|",
              f"| T0 | grid beats - nearest reference onset (quantise target) | {_c(r['T0_grid_beats_minus_reference'])} | |",
              f"| T1 | raw drum detections - reference MIDI | {_c(r['T1_raw_detection_minus_reference'])} | |",
              f"| T2 | score times ({r['quantize_strength']} quantise) - raw detections | {_c(r['T2_score_minus_raw_detection'])} | |",
              f"| T1+T2 | score times - reference MIDI | {_c(r['T1+T2_score_minus_reference'])} | |",
              f"| T3 | page signal path, first audible sample - scheduled (shipped / compressor bypassed) | "
              f"{r['T3_signal_path_shipped_ms']} / {r['T3_signal_path_compressor_bypassed_ms']} | |",
              f"| T4 | detector on synth hits - scheduled | {_c(r['T4_detector_on_synth_hop128']['all'])} | {_c(r['T4_detector_on_synth_hop32']['all'])} |",
              f"| T4 kick / snare / hat | | {' / '.join(_c(r['T4_detector_on_synth_hop128'][k]) for k in ('kick', 'snare', 'hat'))} | "
              f"{' / '.join(_c(r['T4_detector_on_synth_hop32'][k]) for k in ('kick', 'snare', 'hat'))} |",
              f"| T5 | detector on original - reference MIDI | {_c(r['T5_detector_on_original_hop128'])} | {_c(r['T5_detector_on_original_hop32'])} |",
              f"| T6 | **reported: render onsets - original onsets** | {_c(r['T6_render_minus_original_hop128'])} | {_c(r['T6_render_minus_original_hop32'])} |",
              f"| | render onsets - scheduled note times | {_c(r['render_onsets_minus_scheduled_hop128'])} | {_c(r['render_onsets_minus_scheduled_hop32'])} |",
              f"| | predicted T6 = T1+T2 + T4 - T5 | {r['predicted_T6_hop128_ms']} | {r['predicted_T6_hop32_ms']} |", ""]
        m = lambda k: (r[k] or {}).get("median_ms")
        comp = np.median(list(r["T3_signal_path_shipped_ms"].values()))
        byp = np.median(list(r["T3_signal_path_compressor_bypassed_ms"].values()))
        t4, t5 = r["T4_detector_on_synth_hop128"]["all"]["median_ms"], m("T5_detector_on_original_hop128")
        L += [f"**Finding ({r['clip']}, hop 128 medians):** the score itself is {m('T1+T2_score_minus_reference'):+.1f} ms late "
              f"of the reference: the raw detections are {m('T1_raw_detection_minus_reference'):+.1f} ms, and the "
              f"{r['quantize_strength']} soft quantise adds {m('T2_score_minus_raw_detection'):+.1f} ms because the grid it pulls "
              f"towards sits {m('T0_grid_beats_minus_reference'):+.1f} ms after the note-ons. The page then plays every sound "
              f"{comp:.2f} ms after its scheduled time ({byp:.2f} ms with the compressor bypassed: the DynamicsCompressor's "
              f"fixed look-ahead; the scheduler itself is exact). Net of that delay the detector fires {t4 - comp:+.1f} ms after "
              f"a synth hit starts vs {t5:+.1f} ms after a soundfont hit, so the attack shape contributes {t4 - comp - t5:+.1f} ms "
              f"(synth hits are detected sooner, not later). Predicted {r['predicted_T6_hop128_ms']:+.1f} ms vs measured "
              f"{m('T6_render_minus_original_hop128'):+.1f} ms.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    run()
