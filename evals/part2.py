"""Part 2 eval: the brief's chat sessions, headless, through the real LLM.

For every session in the manifest: pick the first clip whose audio exists
(and say which), analyse it, then send each turn to the Gemini agent. Every
turn is recorded (user message -> raw model output -> validated ops -> echo ->
result). After every turn that changed the audio we export the edited audio,
its transformed grid and a click audition, and check
  - format: sample rate, channels, bit depth == source
  - splices: source-referenced click scores (evals/clicks.py), plus a hard-cut
    negative control on the same audio
  - grid truth: beat-track the edited audio from scratch and compare with the
    transformed grid (it is never re-detected inside the product)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

from evals.clicks import splice_scores
from evals.references import load_manifest, resolve_clips


def grid_check(edited_audio: str, grid) -> dict:
    from beat_agent.analysis.pipeline import analyze
    re = analyze(edited_audio)
    est, ref = re.beat_times(), grid.beat_times()
    if not len(est) or not len(ref):
        return {"note": "no beats on one side"}
    f = mir_eval.beat.f_measure(ref, est, 0.07)
    d = np.array([np.min(np.abs(est - t)) for t in ref])
    m = d <= 0.07
    out = {"beat_f_vs_redetected": round(float(f), 4), "matched": int(m.sum()), "of": int(len(ref)),
           "median_abs_ms": round(float(1e3 * np.median(d[m])), 2) if m.any() else None,
           "p95_abs_ms": round(float(1e3 * np.percentile(d[m], 95)), 2) if m.any() else None}
    if grid.downbeats and re.downbeats:
        out["downbeat_f_vs_redetected"] = round(float(mir_eval.beat.f_measure(np.array(grid.downbeats),
                                                                              np.array(re.downbeats), 0.07)), 4)
    out["redetected_bpm"] = re.tempo.bpm
    # If the user corrected the tempo octave, the fresh tracker is at the old level;
    # compare at the grid's level as well (re-detected beats doubled / halved).
    oc = grid.tempo.octave_correction
    if oc != 1.0 and len(est) > 2:
        # phase-free: halved grid -> share of grid beats the fresh tracker confirms;
        # doubled grid -> share of fresh-tracker beats the grid contains
        a, b = (ref, est) if oc < 1 else (est, ref)
        conf = float(np.mean([np.min(np.abs(b - t)) <= 0.07 for t in a]))
        out["beat_f_at_grid_level"] = round(conf, 4)
        out["note"] = (f"user set tempo octave x{oc:g}; the fresh tracker is at its own level, so the "
                       "grid-level number is the share of beats confirmed, not an F-measure")
    return out


def run(manifest_path="evals/manifest.yaml", out_root="evals/out/part2", results_dir="evals/results") -> dict:
    from beat_agent.agent.agent import Agent
    from beat_agent.agent.llm import GeminiProvider, LLMError, UsageLedger
    from beat_agent.analysis.pipeline import analyze
    from beat_agent.edit.session import EditSession
    from beat_agent.edit.timeline import render
    from beat_agent.io.audio import open_source

    out_root, results_dir = Path(out_root), Path(results_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    man = load_manifest(manifest_path)
    clips = {c.id: c for c in resolve_clips(man, Path("evals/out/part1/_generated"))}
    try:
        llm = GeminiProvider()
    except LLMError as e:
        msg = f"Part 2 not run: {e}"
        (results_dir / "part2.md").write_text(f"# Part 2\n\n**{msg}**\n")
        print(msg)
        return {"error": msg}
    total = UsageLedger()
    sessions_out = []
    llm_failure = None
    for s in man.get("sessions", []):
        if llm_failure:
            break
        prefs = s["clip"] if isinstance(s["clip"], list) else [s["clip"]]
        clip = next((clips[c] for c in prefs if c in clips and clips[c].audio), None)
        rec = {"id": s["id"], "requested_clip": prefs[0], "note": s.get("note"), "turns": []}
        if clip is None:
            rec["status"] = "missing"
            sessions_out.append(rec)
            continue
        kind = "test pack" if not (clip.substitute or clip.generated) else ("SUBSTITUTE" if clip.substitute else "generated")
        rec.update(clip=clip.id, clip_kind=kind, clip_desc=clip.desc, audio=clip.audio)
        print(f"=== {s['id']} on {clip.id} ({kind})", flush=True)
        d = out_root / s["id"]
        d.mkdir(parents=True, exist_ok=True)
        src = open_source(clip.audio)
        grid = analyze(clip.audio, src)
        (d / "source.grid.json").write_text(grid.model_dump_json(indent=1))
        sess = EditSession(src, grid)
        agent = Agent(llm, sess)
        src_audio = sf.read(src.readable_path, always_2d=True)[0]
        for k, text in enumerate(s["turns"], 1):
            t0 = time.perf_counter()
            res, tr = agent.handle(text)
            if tr.llm_error:
                llm_failure = f"{s['id']} turn {k}: {tr.llm_error}"
                break
            turn = {"n": k, "user": text, "status": res.status, "echo": res.message,
                    "llm_attempts": [{"raw": a["raw"], "error": a["error"], "cached": a["cached"], "usage": a["usage"],
                                      "fetched_at": a.get("fetched_at"), "cache_key": a.get("cache_key")}
                                     for a in tr.attempts],
                    "ops": tr.plan["ops"] if tr.plan else None, "reading": tr.plan["reading"] if tr.plan else None,
                    "resolved": res.resolved, "length_before_s": res.length_before_s,
                    "length_after_s": res.length_after_s, "seconds": round(time.perf_counter() - t0, 2)}
            if res.status in ("applied", "undo", "redo"):
                stem = f"turn{k}"
                ex = sess.export(d, stem)
                info, si = sf.info(ex["audio"]), src
                turn["files"] = {"audio": ex["audio"], "grid": ex["grid"], "click_audition": ex["click"]}
                turn["format"] = {"samplerate": info.samplerate, "channels": info.channels, "subtype": info.subtype,
                                  "matches_source": (info.samplerate, info.channels, info.subtype) ==
                                                    (si.samplerate, si.channels, si.export_subtype)}
                y = sf.read(ex["audio"], always_2d=True)[0]
                turn["splices"] = splice_scores(y, src_audio, src.samplerate, ex["splice_points"])
                if ex["splice_points"]:
                    hard = d / f"{stem}_HARDCUT_negative_control.wav"
                    render(src, sess.state.timeline, hard, crossfade_s=2 / src.samplerate)
                    turn["splices_hard_cut_control"] = splice_scores(sf.read(hard, always_2d=True)[0], src_audio,
                                                                     src.samplerate, ex["splice_points"])
                turn["grid_check"] = grid_check(ex["audio"], sess.grid())
            rec["turns"].append(turn)
            print(f"  [{res.status}] {text!r}\n    " + res.message.replace("\n", "\n    "), flush=True)
        rec["usage"] = agent.ledger.to_dict()
        for f in ("calls", "cached_calls", "prompt_tokens", "output_tokens", "thinking_tokens"):
            setattr(total, f, getattr(total, f) + getattr(agent.ledger, f))
        (d / "transcript.md").write_text(transcript_md(rec))
        sessions_out.append(rec)
    result = {"model": llm.model, "sessions": sessions_out, "usage_total": total.to_dict(), "cache_inventory": cache_inventory(),
              "llm_provenance": provenance([a for s in sessions_out for t in s["turns"] for a in t["llm_attempts"]],
                                           llm.model)}
    if llm_failure:
        # A run cut short by quota is never published: evals/results keeps the last complete run.
        # Answers that did arrive are cached, so the next run replays them without new calls.
        result["incomplete"] = llm_failure
        (out_root / "part2.partial.json").write_text(json.dumps(result, indent=1, default=str))
        print(f"Part 2 INCOMPLETE ({llm_failure}); evals/results/part2.* left unchanged. "
              f"Partial run in {out_root / 'part2.partial.json'}")
        return result
    (results_dir / "part2.json").write_text(json.dumps(result, indent=1, default=str))
    (results_dir / "part2.md").write_text(report(result))
    return result


def provenance(attempts: list[dict], model: str) -> dict:
    """Where this run's answers came from: live API calls or cached replays, and when they were fetched."""
    from beat_agent.agent.llm import cache_mode
    when = sorted(a["fetched_at"] for a in attempts if a.get("fetched_at"))
    return {"model": model, "cache_mode": cache_mode(), "calls": len(attempts),
            "replayed_from_cache": sum(1 for a in attempts if a.get("cached")),
            "live_calls": sum(1 for a in attempts if not a.get("cached")),
            "fetched_utc_first": when[0] if when else None, "fetched_utc_last": when[-1] if when else None}


def provenance_line(p: dict | None) -> str:
    if not p:
        return "LLM provenance: not recorded."
    src = ("all answers replayed from the local cache" if p["live_calls"] == 0 else
           "all answers fetched live" if p["replayed_from_cache"] == 0 else
           f"{p['replayed_from_cache']} replayed from the cache, {p['live_calls']} fetched live")
    day = lambda s: (s or "?")[:10]
    span = day(p["fetched_utc_first"]) if day(p["fetched_utc_first"]) == day(p["fetched_utc_last"]) else \
        f"{day(p['fetched_utc_first'])} to {day(p['fetched_utc_last'])}"
    return (f"LLM provenance: `{p['model']}`, {p['calls']} calls, {src} (LLM_CACHE={p['cache_mode']}); "
            f"answers originally fetched from the API on {span} (UTC). Re-run live with `LLM_CACHE=off` "
            "or `python -m evals.run part2 --live`.")


def cache_inventory() -> dict:
    """What is in the local LLM answer cache: one file per distinct prompt (model, prompt, schema).

    This is NOT a count of API calls: a prompt answered live twice (LLM_CACHE=off) is one file, calls
    that failed (quota) leave none, and other sessions (the preview app) add files. Live API calls are
    counted per run in `usage_total` (calls - cached_calls)."""
    from beat_agent.agent.llm import price_per_mtok
    pin, pout = price_per_mtok()
    calls = tok = 0
    cost = 0.0
    by_model: dict[str, int] = {}
    from beat_agent.agent.llm import _cache_dir
    for f in _cache_dir().glob("*.json"):
        d = json.loads(f.read_text())
        u = d.get("usage", {})
        calls += 1
        by_model[d.get("model", "?")] = by_model.get(d.get("model", "?"), 0) + 1
        tok += sum(u.values())
        cost += (u.get("prompt_tokens", 0) * pin + (u.get("output_tokens", 0) + u.get("thinking_tokens", 0)) * pout) / 1e6
    return {"unique_cached_responses": calls, "by_model": by_model, "tokens": tok,
            "est_cost_usd_to_produce": round(cost, 5)}


def transcript_md(rec: dict) -> str:
    L = [f"# Session {rec['id']} on `{rec.get('clip')}` ({rec.get('clip_kind')})", ""]
    if rec.get("clip") != rec["requested_clip"]:
        L.append(f"The brief asks for **{rec['requested_clip']}**, which is not available; this session ran on "
                 f"**{rec.get('clip')}** -- {rec.get('clip_desc')}.")
        L.append("")
    if rec.get("note"):
        L += [f"Note: {rec['note']}", ""]
    for t in rec["turns"]:
        L += [f"## Turn {t['n']}: \"{t['user']}\"", "", f"**Status:** `{t['status']}`", ""]
        for i, a in enumerate(t["llm_attempts"], 1):
            L += [f"LLM attempt {i}{' (cached)' if a['cached'] else ''}: tokens {a['usage']}", "",
                  "```json", a["raw"] or "", "```", ""]
            if a["error"]:
                L += ["Validation error:", "```", a["error"], "```", ""]
        L += ["**Echo shown to the user:**", "", "```", t["echo"], "```", ""]
        if t.get("files"):
            L += [f"Files: `{t['files']['audio']}`, grid `{t['files']['grid']}`, click audition `{t['files']['click_audition']}`", ""]
            L += [f"Format: {t['format']}", ""]
            if t.get("splices"):
                L.append("Splices (step ratio > 1 or audible HF excess = click; see evals/clicks.py):")
                for sp, hc in zip(t["splices"], t.get("splices_hard_cut_control", [{}] * len(t["splices"]))):
                    L.append(f"- at {sp['splice_s']} s: step {sp['step_ratio']}, zone jump {sp['jump_ratio']}, HF {sp['hf_ratio']} "
                             f"({sp['hf_rel_db']} dB) -> {'CLICK' if sp['flagged'] else 'clean'}; "
                             f"hard-cut control: step {hc.get('step_ratio')} -> "
                             f"{'CLICK' if hc.get('flagged') else 'not a click either (waveforms agree there)'}")
                L.append("")
            L += [f"Grid check (edited audio re-tracked vs transformed grid): {t['grid_check']}", ""]
    L += [f"Token usage: {rec.get('usage')}", ""]
    return "\n".join(L)


def _gc_cell(g: dict) -> str:
    v = str(g.get("beat_f_vs_redetected"))
    if "beat_f_at_grid_level" in g:
        v += f" ({g['beat_f_at_grid_level']} at grid level)"
    return v


def report(res: dict) -> str:
    L = ["# Part 2 results: chat editing", "",
         f"Generated by `python -m evals.run part2`. Model: `{res['model']}` (real API; responses cached on disk "
         "for re-runs, cached calls cost nothing).", "",
         f"**{provenance_line(res.get('llm_provenance'))}**", "",
         "Click columns: our crossfaded splices vs the same edit rendered with a hard cut (negative control). A hard cut is only",
         "a click where the two waveforms disagree, so the control is not flagged everywhere; where it is, the detector demonstrably",
         "sees clicks that the crossfade removed.", "",
         "| session | ran on | kind | turns | statuses | splices (clicks / total) | hard-cut control flagged | "
         "grid check: beat F vs re-detected | median abs ms | formats match |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["sessions"]:
        if r.get("status") == "missing":
            L.append(f"| {r['id']} | **missing** | - | - | - | - | - | - | - | - |")
            continue
        sp = [x for t in r["turns"] for x in t.get("splices", [])]
        hc = [x for t in r["turns"] for x in t.get("splices_hard_cut_control", [])]
        gc = [t["grid_check"] for t in r["turns"] if t.get("grid_check")]
        fm = [t["format"]["matches_source"] for t in r["turns"] if t.get("format")]
        L.append(f"| {r['id']} | {r['clip']} | {r['clip_kind']} | {len(r['turns'])} | "
                 f"{', '.join(t['status'] for t in r['turns'])} | {sum(x['flagged'] for x in sp)} / {len(sp)} | "
                 f"{sum(x['flagged'] for x in hc)} / {len(hc)} | "
                 f"{', '.join(_gc_cell(g) for g in gc) or '-'} | "
                 f"{', '.join(str(g.get('median_abs_ms')) for g in gc) or '-'} | {all(fm) if fm else '-'} |")
    u = res["usage_total"]
    ci = res.get("cache_inventory") or {}
    price = (f"at an assumed {u['price_assumption_usd_per_mtok']['input']} / "
             f"{u['price_assumption_usd_per_mtok']['output_incl_thinking']} USD per 1M tokens (upper bound: the highest "
             "listed Gemini 3 Flash price; the free tier charged $0)")
    L += ["", "## LLM usage", "",
          f"**This run: {u['calls'] - u['cached_calls']} live API calls**, {u['cached_calls']} answers replayed from the cache "
          f"({u['calls']} turns' worth of model answers in total). Live calls used {u['prompt_tokens']} prompt + "
          f"{u['output_tokens']} output + {u['thinking_tokens']} thinking tokens; estimated cost "
          f"**${u['est_cost_usd']:.4f}** {price}.", "",
          f"Local cache inventory (not a call count): **{ci.get('unique_cached_responses')} unique cached LLM responses** "
          f"({', '.join(f'{k}: {v}' for k, v in (ci.get('by_model') or {}).items())}), one per distinct prompt; producing "
          f"them took at least that many live calls, {ci.get('tokens')} tokens, estimated "
          f"${ci.get('est_cost_usd_to_produce', 0):.4f} at the same upper-bound price (budget: $10).", "",
          "## Transcripts", ""]
    for r in res["sessions"]:
        if r.get("status") != "missing":
            L.append(f"- [{r['id']}](../out/part2/{r['id']}/transcript.md) on {r['clip']} ({r['clip_kind']})")
    L += ["", "## Turn by turn", ""]
    for r in res["sessions"]:
        if r.get("status") == "missing":
            continue
        for t in r["turns"]:
            L += [f"**{r['id']}.{t['n']}** \"{t['user']}\" -> `{t['status']}`  ", f"ops: `{json.dumps(t['ops'])}`", "",
                  "```", t["echo"], "```", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    run()
