"""Part 1 eval: beat grid accuracy, drift, sync, speed.

Every number in evals/results/part1.md is computed here from files this run
wrote. Clips without a reference get a grid + click audition and "not
measured" for accuracy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mir_eval  # noqa: E402
import numpy as np  # noqa: E402

from evals import conventions as C  # noqa: E402
from evals.references import Clip, load_manifest, resolve_clips  # noqa: E402

NM = "not measured"


def _f(ref, est, trim):
    if ref is None or not len(ref):
        return None
    if trim:
        ref, est = mir_eval.beat.trim_beats(np.asarray(ref)), mir_eval.beat.trim_beats(np.asarray(est))
    if not len(est):
        return 0.0
    return round(float(mir_eval.beat.f_measure(np.asarray(ref), np.asarray(est), C.BEAT_F_WINDOW_S)), 4)


def drift(ref: np.ndarray, est: np.ndarray) -> dict:
    """Signed error of the nearest grid beat for every reference beat (within 70 ms)."""
    if ref is None or not len(ref) or not len(est):
        return {}
    j = np.clip(np.searchsorted(est, ref), 1, len(est) - 1)
    near = np.where(np.abs(est[j - 1] - ref) < np.abs(est[j] - ref), est[j - 1], est[j])
    err = near - ref
    m = np.abs(err) <= C.BEAT_F_WINDOW_S
    if not m.any():
        return {"matched": 0}
    e = err[m]
    return {"matched": int(m.sum()), "of": int(len(ref)), "max_abs_ms": round(float(1e3 * np.abs(e).max()), 2),
            "mean_abs_ms": round(float(1e3 * np.abs(e).mean()), 2), "mean_signed_ms": round(float(1e3 * e.mean()), 2),
            "within_10ms": round(float((np.abs(e) <= 0.010).mean()), 4),
            "_t": ref[m].tolist(), "_e": (1e3 * e).tolist()}


def librosa_baseline(path) -> np.ndarray:
    import librosa
    from beat_agent.io.audio import load_analysis_mono, open_source
    y = load_analysis_mono(open_source(path))
    _, beats = librosa.beat.beat_track(y=y, sr=22050, units="time")
    return np.asarray(beats)


def plots(clip: Clip, grid, d: dict, out: Path, y=None):
    est = grid.beat_times()
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    if d.get("_t"):
        ax[0].scatter(d["_t"], d["_e"], s=4)
        ax[0].axhspan(-10, 10, color="0.9", zorder=-1)
        ax[0].set_title(f"{clip.id}: grid - reference per beat (ms) vs time; band = +-10 ms")
        ax[0].set_xlabel("s")
        ax[1].hist(d["_e"], bins=60)
        ax[1].set_title("error histogram (ms)")
    else:
        ax[0].text(0.5, 0.5, "no reference: drift not measured", ha="center")
        ax[1].text(0.5, 0.5, "no reference", ha="center")
    # onset envelope vs grid over a few seconds where the beat starts
    from beat_agent.analysis.onsets import spectral_flux
    from beat_agent.io.audio import load_analysis_mono, open_source
    if y is None:
        y = load_analysis_mono(open_source(clip.audio))
    fl, fps = spectral_flux(y, 22050, 0.005)
    t0 = float(est[0]) - 1.0 if len(est) else 0.0
    t1 = t0 + 8.0
    tt = np.arange(len(fl)) / fps
    m = (tt >= t0) & (tt <= t1)
    ax[2].plot(tt[m], fl[m], lw=0.7, label="spectral flux")
    for b in est[(est >= t0) & (est <= t1)]:
        ax[2].axvline(b, color="tab:red", lw=0.8)
    if clip.beats is not None:
        for b in clip.beats[(clip.beats >= t0) & (clip.beats <= t1)]:
            ax[2].axvline(b, color="tab:green", lw=0.8, ls="--")
    ax[2].set_title("onset envelope; red = grid, green dashed = reference")
    fig.tight_layout()
    fig.savefig(out / "grid_diagnostics.png", dpi=90)
    plt.close(fig)


def run(manifest_path="evals/manifest.yaml", out_root="evals/out/part1", results_dir="evals/results") -> dict:
    from beat_agent.analysis.pipeline import analyze
    from beat_agent.io.audio import open_source
    from beat_agent.io.audition import write_click_audition
    from beat_agent.sync.sync import SyncError, render

    out_root, results_dir = Path(out_root), Path(results_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    man = load_manifest(manifest_path)
    clips = [c for c in resolve_clips(man, out_root / "_generated") if not c.part3]
    rows, grids, srcs = [], {}, {}
    for clip in clips:
        row = {"id": clip.id, "desc": clip.desc, "generated": clip.generated, "substitute": clip.substitute, "audio": clip.audio,
               "reference": clip.ref_files or ("generated truth" if clip.generated else None)}
        if not clip.audio:
            row["status"] = "missing"
            rows.append(row)
            print(f"[missing] {clip.id}: no audio matched the manifest patterns", flush=True)
            continue
        d = out_root / clip.id.replace(":", "_")
        d.mkdir(parents=True, exist_ok=True)
        src = open_source(clip.audio)
        t0 = time.perf_counter()
        grid = analyze(clip.audio, src)
        runtime = time.perf_counter() - t0
        grids[clip.id], srcs[clip.id] = grid, src
        (d / "grid.json").write_text(grid.model_dump_json(indent=1))
        write_click_audition(src, grid, d / "click_audition.wav")
        est = grid.beat_times()
        ts = grid.time_signature
        row.update(status="ok", duration_s=round(src.duration, 2), runtime_s=round(runtime, 2),
                   bpm=grid.tempo.bpm, tempo_type=grid.tempo.type,
                   meter=f"{ts.numerator}/{ts.denominator}" if ts.numerator else None,
                   n_beats=len(est), tempo_changes=grid.tempo.changes, messages=grid.messages,
                   confidence=grid.confidence.model_dump(exclude={"how"}))
        has_ref = clip.beats is not None and len(clip.beats) > 0
        if has_ref:
            row["beat_f"] = _f(clip.beats, est, trim=True)
            row["beat_f_untrimmed"] = _f(clip.beats, est, trim=False)
            row["downbeat_f"] = _f(clip.downbeats, np.array(grid.downbeats), trim=True) if clip.downbeats is not None else NM
            base = librosa_baseline(clip.audio)
            row["librosa_beat_f"] = _f(clip.beats, base, trim=True)
            abl = analyze(clip.audio, src, phase_refinement=False)
            row["beat_f_no_phase_refinement"] = _f(clip.beats, abl.beat_times(), trim=True)
            dr = drift(clip.beats, est)
            dr_abl = drift(clip.beats, abl.beat_times())
            row["drift"] = {k: v for k, v in dr.items() if not k.startswith("_")}
            row["drift_no_phase_refinement"] = {k: v for k, v in dr_abl.items() if not k.startswith("_")}
        else:
            row.update(beat_f=NM, downbeat_f=NM, librosa_beat_f=NM, drift=NM)
            dr = {}
        plots(clip, grid, dr, d)
        (d / "metrics.json").write_text(json.dumps(row, indent=1, default=str))
        rows.append(row)
        print(f"[ok] {clip.id}: F={row.get('beat_f')} Fdb={row.get('downbeat_f')} drift={row.get('drift')} "
              f"t={runtime:.1f}s", flush=True)

    sync_rows = []
    for s in man.get("sync", []):
        a, b = s["master"], s["other"]
        r = {"pair": f"{b} -> {a} at {a} bar {s['a_bar']}", "generated": bool(s.get("generated"))}
        if a not in grids or b not in grids:
            r["status"] = "missing"
        else:
            try:
                res = render(srcs[a], grids[a], srcs[b], grids[b], out_root / f"sync_{b}_to_{a}",
                             a_bar=s["a_bar"], b_bar=s.get("b_bar", 1))
                k = dict(res["kick_alignment"])
                k.pop("offsets_ms", None)
                r.update(status="ok", plan=res["plan"], kick_alignment=k)
            except SyncError as e:
                r.update(status="error", error=str(e))
        sync_rows.append(r)
        print(f"[sync] {r}", flush=True)

    result = {"clips": rows, "sync": sync_rows}
    (results_dir / "part1.json").write_text(json.dumps(result, indent=1, default=str))
    (results_dir / "part1.md").write_text(report(result))
    return result


def _v(x):
    return "-" if x is None else x


def report(res: dict) -> str:
    L = ["# Part 1 results: beat grid, drift, sync, speed", "",
         "Generated by `python -m evals.run part1`. Every number below comes from files in `evals/out/part1/`.",
         "Three kinds of rows, never mixed in a claim: **test pack** (the official clips; `missing` when absent at run time),",
         "**SUBSTITUTE (GTZAN)** (real music with public human annotations, standing in while the test pack is missing;",
         "GTZAN is held out from the Beat This! checkpoint used), and **generated** (rendered by `evals/synth.py`, exact truth).",
         "Substitute annotations come from a different annotation effort than the test pack and have their own small timing",
         "offsets, so drift numbers on them measure grid + annotation disagreement.", "",
         "Beat/downbeat F: mir_eval, +-70 ms, first 5 s trimmed (mir_eval default). `untrimmed` scores from 0 s.", "",
         "| clip | source | status | beat F | untrimmed | no phase refinement | librosa F | downbeat F | BPM | type | meter | runtime s / audio s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = lambda r: (2 if r.get("generated") else 1 if r.get("substitute") else 0)
    for r in sorted(res["clips"], key=order):
        src = ("generated" if r.get("generated") else "SUBSTITUTE (GTZAN)" if r.get("substitute")
               else "user" if str(r["id"]).startswith("user:") else "test pack")
        if r["status"] != "ok":
            L.append(f"| {r['id']} | {src} | **{r['status']}** | - | - | - | - | - | - | - | - | - |")
            continue
        L.append(f"| {r['id']} | {src} | ok | {_v(r.get('beat_f'))} | {_v(r.get('beat_f_untrimmed', NM))} | "
                 f"{_v(r.get('beat_f_no_phase_refinement', NM))} | {_v(r.get('librosa_beat_f'))} | {_v(r.get('downbeat_f'))} | "
                 f"{r['bpm']} | {r['tempo_type']} | {r['meter']} | {r['runtime_s']} / {r['duration_s']} |")
    L += ["", "## Drift (grid beat - reference beat, every matched reference beat)", "",
          "| clip | matched | max abs ms | mean abs ms | mean signed ms | share within 10 ms | max abs ms without phase refinement |",
          "|---|---|---|---|---|---|---|"]
    for r in res["clips"]:
        d = r.get("drift")
        if isinstance(d, dict) and d.get("matched"):
            da = r.get("drift_no_phase_refinement", {})
            L.append(f"| {r['id']} | {d['matched']}/{d['of']} | {d['max_abs_ms']} | {d['mean_abs_ms']} | {d['mean_signed_ms']} | "
                     f"{d['within_10ms']} | {da.get('max_abs_ms', '-')} |")
    L += ["", "## Sync (kick onsets of A vs stretched B in the rendered overlap)", "",
          "| pair | source | status | B rate | offset s | kick pairs | median abs ms | max abs ms | p95 ms |", "|---|---|---|---|---|---|---|---|---|"]
    for s in res["sync"]:
        src = "generated" if s.get("generated") else "test pack"
        if s["status"] != "ok":
            L.append(f"| {s['pair']} | {src} | **{s['status']}** {s.get('error', '')} | - | - | - | - | - | - |")
            continue
        k, p = s["kick_alignment"], s["plan"]
        L.append(f"| {s['pair']} | {src} | ok | {p['rate']} | {p['offset_s']} | {k.get('n_pairs')} | "
                 f"{k.get('median_abs_offset_ms', '-')} | {k.get('max_abs_offset_ms', '-')} | {k.get('p95_abs_offset_ms', '-')} |")
    L += ["", "## Messages emitted by the analyser", ""]
    for r in res["clips"]:
        for m in r.get("messages") or []:
            L.append(f"- **{r['id']}**: {m}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    run()
