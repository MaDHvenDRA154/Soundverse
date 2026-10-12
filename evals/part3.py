"""Part 3 eval: audio -> score -> Web Audio page -> offline render, measured.

Clips: the test pack's P1-P4 when present, else the SUBSTITUTE clips built
by evals/substitutes/fetch_midi.py (public MIDI rendered with a GM soundfont).
Per clip:
  - note-level onset-only F (mir_eval.transcription, +-50 ms, offsets ignored)
    against the reference MIDI: drums by class (kick/snare/hat/other), pitched
    notes by pitch, plus drum onsets ignoring class ("rhythm first")
  - onset timing of the offline render vs the original audio
  - page safety scan; network requests during the render fail the run
  - a 2-loop render (seamless looping can be heard and measured at the seam)
P4-type clips (no reference) get onset timing and a factual description.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

from evals import conventions as C

DRUM_CLASS = {35: "kick", 36: "kick", 37: "snare", 38: "snare", 39: "snare", 40: "snare",
              42: "hat", 44: "hat", 46: "hat"}
CLASS_HZ = {"kick": 100.0, "snare": 200.0, "hat": 400.0, "other": 800.0}   # stand-in "pitches" for mir_eval


def ref_notes(midi_path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    drums, pitched = [], []
    for inst in pm.instruments:
        for n in inst.notes:
            if inst.is_drum:
                drums.append((n.start, max(n.end, n.start + 0.05), DRUM_CLASS.get(n.pitch, "other")))
            else:
                pitched.append((n.start, max(n.end, n.start + 0.05), n.pitch))
    return drums, pitched


def _f(ref, est, hz_ref, hz_est):
    if not len(ref):
        return None
    if not len(est):
        return {"f": 0.0, "p": 0.0, "r": 0.0, "n_ref": len(ref), "n_est": 0}
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        np.array([[a, b] for a, b, _ in ref]), np.array(hz_ref), np.array([[a, b] for a, b, _ in est]), np.array(hz_est),
        onset_tolerance=C.NOTE_ONSET_WINDOW_S, offset_ratio=None)
    return {"f": round(float(f), 4), "p": round(float(p), 4), "r": round(float(r), 4), "n_ref": len(ref), "n_est": len(est)}


def note_metrics(score, midi_path) -> dict:
    import librosa
    rd, rp = ref_notes(midi_path)
    t0 = score.clip["start_s"]
    ed, ep = [], []
    for t in score.tracks:
        for n in t.notes:
            s = n.start_s - t0 if score.clip.get("mode") == "bars" else n.start_s
            if t.role == "drums":
                ed.append((s, s + max(0.05, n.dur_s), n.drum))
            else:
                ep.append((s, s + max(0.05, n.dur_s), n.pitch))
    out = {"drums_by_class": _f(rd, ed, [CLASS_HZ[c] for *_, c in rd], [CLASS_HZ[c] for *_, c in ed]),
           "drum_onsets_any_class": _f(rd, ed, [100.0] * len(rd), [100.0] * len(ed)),
           "pitched": _f(rp, ep, list(librosa.midi_to_hz([p for *_, p in rp])), list(librosa.midi_to_hz([p for *_, p in ep])))}
    allr = [(a, b, CLASS_HZ[c]) for a, b, c in rd] + [(a, b, float(librosa.midi_to_hz(p))) for a, b, p in rp]
    alle = [(a, b, CLASS_HZ[c]) for a, b, c in ed] + [(a, b, float(librosa.midi_to_hz(p))) for a, b, p in ep]
    out["all_notes"] = _f(allr, alle, [h for *_, h in allr], [h for *_, h in alle])
    return out


def onset_timing(original_wav, render_wav, t0: float) -> dict:
    import librosa
    y0, sr0 = librosa.load(str(original_wav), sr=22050, mono=True)
    y1, _ = librosa.load(str(render_wav), sr=22050, mono=True)
    # hop 128 (5.8 ms): the default 512-sample hop quantises onsets to 23 ms, which made
    # matching frames report exactly 0.0 ms
    o0 = librosa.onset.onset_detect(y=y0, sr=22050, units="time", backtrack=False, hop_length=128)
    o1 = librosa.onset.onset_detect(y=y1, sr=22050, units="time", backtrack=False, hop_length=128) + t0
    if not len(o0) or not len(o1):
        return {"note": "no onsets"}
    f = mir_eval.onset.f_measure(o0, o1, window=C.NOTE_ONSET_WINDOW_S)[0]
    d = np.array([o1[np.argmin(np.abs(o1 - t))] - t for t in o0])
    m = np.abs(d) <= C.NOTE_ONSET_WINDOW_S
    return {"onset_f_50ms": round(float(f), 4), "n_onsets_original": int(len(o0)), "n_onsets_render": int(len(o1)),
            "median_abs_offset_ms": round(float(1e3 * np.median(np.abs(d[m]))), 2) if m.any() else None,
            "median_signed_offset_ms": round(float(1e3 * np.median(d[m])), 2) if m.any() else None}


def seam_check(render2_wav, loop_s) -> dict:
    """Two loops rendered back to back: the step at the loop seam vs the signal's typical steps."""
    y, sr = sf.read(str(render2_wav), always_2d=True)
    k = int(round(loop_s * sr))
    if k <= 1 or k >= len(y):
        return {}
    step = float(np.abs(y[k] - y[k - 1]).max())
    typ = float(np.percentile(np.abs(np.diff(y[max(0, k - sr // 10):k + sr // 10], axis=0)), 99))
    return {"seam_s": round(k / sr, 4), "seam_step_ratio": round(step / (typ + 1e-9), 3)}


def clips(manifest) -> list[dict]:
    from evals.references import first_audio, first_match
    out = []
    for c in manifest.get("clips", []):
        if c.get("part3"):
            a = first_audio(c.get("audio"))
            if a:
                out.append({"id": c["id"], "kind": "test pack", "audio": a, "midi": first_match(c.get("midi")), "desc": c.get("desc")})
    have = {c["id"] for c in out}
    for c in manifest.get("part3_substitutes", []):
        if c.get("stands_in_for") in have:
            continue
        a = first_audio(c.get("audio"))
        if a:
            out.append({"id": c["id"], "kind": "SUBSTITUTE", "audio": a, "midi": first_match(c.get("midi")) if c.get("reference") else None,
                        "desc": c.get("desc"), "stands_in_for": c.get("stands_in_for")})
        else:
            out.append({"id": c["id"], "kind": "SUBSTITUTE", "audio": None, "desc": c.get("desc"),
                        "stands_in_for": c.get("stands_in_for")})
    return out


def run(manifest_path="evals/manifest.yaml", out_root="evals/out/part3", results_dir="evals/results") -> dict:
    from beat_agent.render.html import render_html, safety_violations, write_html
    from beat_agent.render.offline import render_offline
    from beat_agent.transcribe.pipeline import RecreateError, recreate
    from evals.references import load_manifest
    man = load_manifest(manifest_path)
    out_root, results_dir = Path(out_root), Path(results_dir)
    rows = []
    for c in clips(man):
        row = dict(c)
        if not c.get("audio"):
            row["status"] = "missing (run python -m evals.substitutes.fetch_midi)"
            rows.append(row)
            continue
        d = out_root / c["id"]
        d.mkdir(parents=True, exist_ok=True)
        t_start = time.perf_counter()
        try:
            score = recreate(c["audio"], stems_dir=d / "stems")
        except RecreateError as e:
            row.update(status="error", error=str(e))
            rows.append(row)
            continue
        row["recreate_s"] = round(time.perf_counter() - t_start, 2)
        (d / "score.json").write_text(score.model_dump_json(indent=1))
        html = render_html(score)
        page = write_html(score, d / "page.html")
        row["safety_violations"] = safety_violations(html)
        row["page_bytes"] = len(html.encode())
        render_offline(page, d / "render_1loop.wav", loops=1)
        render_offline(page, d / "render_2loops.wav", loops=2)
        row.update(status="ok", tempo_bpm=score.tempo_bpm, loop_s=round(score.loop_seconds(), 4), bars=score.bars,
                   meter=list(score.time_signature), mode=score.clip["mode"], loop_note=score.provenance["loop"],
                   tracks={t.id: len(t.notes) for t in score.tracks},
                   onset_timing=onset_timing(c["audio"], d / "render_1loop.wav", score.clip["start_s"]),
                   seam=seam_check(d / "render_2loops.wav", score.loop_seconds()))
        if c.get("midi"):
            row["notes"] = note_metrics(score, c["midi"])
        rows.append(row)
        print(f"[ok] {c['id']}: {row.get('notes')} timing={row['onset_timing']}", flush=True)
    chat = chat_demo(man, out_root)
    result = {"clips": rows, "chat": chat}
    results_dir.mkdir(parents=True, exist_ok=True)
    if chat.get("status") == "incomplete":
        # Same rule as Part 2: no partial chat results in evals/results.
        (out_root / "part3.partial.json").write_text(json.dumps(result, indent=1, default=str))
        print(f"Part 3 chat {chat['status']} ({chat.get('error', '')[:200]}); evals/results/part3.* left unchanged.")
        return result
    (results_dir / "part3.json").write_text(json.dumps(result, indent=1, default=str))
    (results_dir / "part3.md").write_text(report(result))
    return result


def chat_demo(man, out_root: Path) -> dict:
    """The brief's three chat changes on a recreation, through the real LLM."""
    spec = man.get("part3_chat")
    if not spec:
        return {}
    from beat_agent.agent.llm import GeminiProvider, LLMError
    from beat_agent.agent.score_agent import ScoreAgent
    from beat_agent.render.html import write_html
    from beat_agent.transcribe.score import Score
    p = out_root / spec["clip"] / "score.json"
    if not p.exists():
        return {"status": "missing", "clip": spec["clip"]}
    try:
        llm = GeminiProvider()
    except LLMError as e:
        return {"status": "not run", "error": str(e)}
    ag = ScoreAgent(llm, Score.model_validate_json(p.read_text()))
    turns = []
    for k, text in enumerate(spec["turns"], 1):
        status, echo, plan = ag.handle(text)
        if ag.llm_error:
            return {"clip": spec["clip"], "status": "incomplete", "error": ag.llm_error, "turns": turns,
                    "usage": ag.ledger.to_dict()}
        page = write_html(ag.score, out_root / spec["clip"] / f"chat_turn{k}.html")
        turns.append({"user": text, "status": status, "ops": plan["ops"] if plan else None, "echo": echo, "page": page})
        print(f"  [{status}] {text!r}: {echo}", flush=True)
    return {"clip": spec["clip"], "turns": turns, "usage": ag.ledger.to_dict()}


def _g(x, k="f"):
    return "-" if not x else x.get(k)


def report(res) -> str:
    L = ["# Part 3 results: audio -> score -> Web Audio page", "",
         "Generated by `python -m evals.run part3`. **SUBSTITUTE** rows are public MIDI excerpts rendered with a General",
         "MIDI soundfont (TimGM6mb) -- not the test pack; the excerpt MIDI is the reference. Note F: mir_eval transcription,",
         "onset-only, +-50 ms. Drums are compared by class (kick / snare / hat / other); 'drum onsets' ignores the class.", "",
         "| clip | kind | stands in for | status | tempo | loop s | tracks (notes) | drums F (by class) | drum onsets F | pitched F | all notes F | render onset F | render median abs ms | seam step ratio | page safety |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["clips"]:
        if r.get("status") != "ok":
            L.append(f"| {r['id']} | {r['kind']} | {r.get('stands_in_for', '-')} | **{r.get('status')}** {r.get('error', '')} | | | | | | | | | | | |")
            continue
        n = r.get("notes") or {}
        ot = r["onset_timing"]
        L.append(f"| {r['id']} | {r['kind']} | {r.get('stands_in_for', '-')} | ok | {r['tempo_bpm']} | {r['loop_s']} | "
                 f"{', '.join(f'{k} ({v})' for k, v in r['tracks'].items())} | {_g(n.get('drums_by_class'))} | "
                 f"{_g(n.get('drum_onsets_any_class'))} | {_g(n.get('pitched'))} | {_g(n.get('all_notes'))} | "
                 f"{ot.get('onset_f_50ms', '-')} | {ot.get('median_abs_offset_ms', '-')} | {r['seam'].get('seam_step_ratio', '-')} | "
                 f"{'ok' if not r['safety_violations'] else r['safety_violations']} |")
    L += ["", "Precision / recall per clip:", ""]
    for r in res["clips"]:
        if r.get("notes"):
            for k, v in r["notes"].items():
                if v:
                    L.append(f"- {r['id']} {k}: P {v['p']}, R {v['r']}, F {v['f']} ({v['n_est']} est / {v['n_ref']} ref)")
    ch = res.get("chat") or {}
    if ch.get("turns"):
        L += ["", f"## Chat changes on the recreation of {ch['clip']} (real LLM)", ""]
        for t in ch["turns"]:
            L += [f"- \"{t['user']}\" -> `{t['status']}` ops `{json.dumps(t['ops'])}`  ", f"  {t['echo'].replace(chr(10), ' / ')}"]
        L.append(f"\nLLM usage: {ch.get('usage')}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    run()
