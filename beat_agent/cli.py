"""beat-agent command line.

  beat-agent analyze <file> [--out DIR]
  beat-agent sync <master> <other> --at-bar N [--b-bar M] [--out DIR]
  beat-agent chat <file> [--out DIR]      interactive editing; :export, :quit
  beat-agent recreate <file> [--bars N] [--start-bar M] [--out DIR] [--render]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _out_dir(arg: str | None, path: str) -> Path:
    d = Path(arg) if arg else Path("out") / Path(path).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_analyze(a) -> int:
    from beat_agent.analysis.pipeline import analyze
    from beat_agent.io.audio import open_source
    from beat_agent.io.audition import write_click_audition
    t0 = time.perf_counter()
    src = open_source(a.file)
    grid = analyze(a.file, src)
    out = _out_dir(a.out, a.file)
    stem = Path(a.file).stem
    (out / f"{stem}.grid.json").write_text(grid.model_dump_json(indent=1))
    aud = write_click_audition(src, grid, out / f"{stem}.click.wav")
    ts = grid.time_signature
    print(f"{Path(a.file).name}: {src.duration:.1f}s, {src.samplerate} Hz, {src.channels} ch, {src.export_subtype}")
    if grid.beats:
        print(f"  tempo {grid.tempo.bpm} BPM ({grid.tempo.type}), {len(grid.beats)} beats, "
              f"{grid.n_bars()} bars, {ts.numerator}/{ts.denominator}" if ts.numerator else
              f"  tempo {grid.tempo.bpm} BPM ({grid.tempo.type}), {len(grid.beats)} beats, meter unknown")
        for c in grid.tempo.changes:
            print(f"  tempo change at beat {c['beat_index']} (bar {c['bar']}, {c['time']:.2f}s): {c['from_bpm']} -> {c['to_bpm']} BPM")
        alts = ", ".join(f"x{t.factor:g}={t.bpm:.1f} (support {t.onset_support:.2f})" for t in grid.tempo.alternatives)
        print(f"  tempo octave candidates: {alts}")
        print(f"  confidence overall {grid.confidence.overall}, downbeat {grid.confidence.downbeat}, coverage {grid.confidence.coverage}")
    for m in grid.messages:
        print(f"  note: {m}")
    print(f"  wrote {out / (stem + '.grid.json')} and {aud['path']}  ({time.perf_counter() - t0:.1f}s)")
    return 0


def cmd_sync(a) -> int:
    from beat_agent.analysis.pipeline import analyze
    from beat_agent.io.audio import open_source
    from beat_agent.sync.sync import render
    sa, sb = open_source(a.master), open_source(a.other)
    ga, gb = analyze(a.master, sa), analyze(a.other, sb)
    out = _out_dir(a.out, f"sync_{Path(a.master).stem}_{Path(a.other).stem}")
    r = render(sa, ga, sb, gb, out, a_bar=a.at_bar, b_bar=a.b_bar, overlap_s=a.seconds)
    p, k = r["plan"], r["kick_alignment"]
    print(f"B bar {p['b_bar']} -> A bar {p['a_bar']} ({p['a_start_s']:.3f}s). B rate x{p['rate']:.4f} "
          f"(stretch {p['stretch_ratio']:.4f}), offset {p['offset_s']:+.3f}s, {p['n_mapped_beats']} beats mapped.")
    for w in p["warnings"]:
        print(f"  warning: {w}")
    if k.get("n_pairs"):
        print(f"  kick alignment: median {k['median_abs_offset_ms']} ms, max {k['max_abs_offset_ms']} ms over {k['n_pairs']} kick pairs")
    else:
        print(f"  kick alignment not measured: {k.get('note', 'no kick pairs')}")
    print(f"  wrote {out}")
    return 0


def cmd_chat(a) -> int:
    from beat_agent.agent.agent import Agent
    from beat_agent.agent.llm import GeminiProvider, LLMError
    from beat_agent.analysis.pipeline import analyze
    from beat_agent.edit.session import EditSession
    from beat_agent.io.audio import open_source
    try:
        provider = GeminiProvider()   # live calls with your key; no cache (LLM_CACHE is for the eval harness)
    except LLMError as e:
        print(e, file=sys.stderr)
        return 2
    src = open_source(a.file)
    print(f"analysing {a.file} ...", flush=True)
    sess = EditSession(src, analyze(a.file, src))
    agent = Agent(provider, sess)
    out = _out_dir(a.out, a.file)
    print(sess.grid_header())
    print("Type an edit (beats/bars), 'undo', 'redo', ':export' to write files, ':quit' to leave.")
    n = 0
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in (":q", ":quit", "quit", "exit"):
            break
        if text == ":export":
            n += 1
            ex = sess.export(out, f"edit{n:02d}")
            print(f"wrote {ex['audio']}, {ex['grid']}, {ex['click']}")
            continue
        res, _ = agent.handle(text)
        print(f"[{res.status}] {res.message}")
    u = agent.ledger.to_dict()
    print(f"LLM: {u['calls']} calls, {u['prompt_tokens']} in / {u['output_tokens'] + u['thinking_tokens']} out tokens, "
          f"est. ${u['est_cost_usd']}")
    return 0


def cmd_recreate(a) -> int:
    from beat_agent.render.html import safety_violations, write_html, render_html
    from beat_agent.transcribe.pipeline import recreate
    out = _out_dir(a.out, a.file)
    stem = Path(a.file).stem
    t0 = time.perf_counter()
    score = recreate(a.file, bars=a.bars, start_bar=a.start_bar, stems_dir=out / "stems" if a.stems else None)
    (out / f"{stem}.score.json").write_text(score.model_dump_json(indent=1))
    page = write_html(score, out / f"{stem}.html")
    bad = safety_violations(render_html(score))
    c = score.clip
    print(f"{score.bars} bars ({c['mode']}: {c['start_s']:.3f}-{c['end_s']:.3f} s), {score.tempo_bpm} BPM, "
          f"{score.time_signature[0]}/{score.time_signature[1]}; tracks: "
          + ", ".join(f"{t.name} {len(t.notes)} notes" for t in score.tracks) + f"  ({time.perf_counter() - t0:.1f}s)")
    print(f"  wrote {out / (stem + '.score.json')} and {page}" + (f"  SAFETY: {bad}" if bad else ""))
    if a.render:
        from beat_agent.render.offline import render_offline
        w = render_offline(page, out / f"{stem}.render.wav", loops=a.loops)
        print(f"  offline render: {w}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="beat-agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze", help="beat grid JSON + click audition")
    p.add_argument("file")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("sync", help="beat-match <other> to <master>")
    p.add_argument("master")
    p.add_argument("other")
    p.add_argument("--at-bar", type=int, required=True, help="A's bar (1-based) where the overlap starts")
    p.add_argument("--b-bar", type=int, default=1, help="B's bar (1-based) placed on A's --at-bar (default 1)")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_sync)
    p = sub.add_parser("chat", help="interactive chat editing of <file>")
    p.add_argument("file")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_chat)
    p = sub.add_parser("recreate", help="audio clip -> score JSON + self-contained Web Audio page")
    p.add_argument("file")
    p.add_argument("--bars", type=int, help="number of bars (1-16); default: the whole clip, or 8 bars of a long track")
    p.add_argument("--start-bar", type=int)
    p.add_argument("--out")
    p.add_argument("--render", action="store_true", help="also render the page offline to WAV (headless Chromium)")
    p.add_argument("--loops", type=int, default=1)
    p.add_argument("--stems", action="store_true", help="keep the Demucs stems")
    p.set_defaults(fn=cmd_recreate)
    a = ap.parse_args(argv)
    from beat_agent.io.audio import AudioInputError
    from beat_agent.agent.llm import LLMError
    from beat_agent.sync.sync import SyncError
    from beat_agent.transcribe.pipeline import RecreateError
    try:
        return a.fn(a)
    except (AudioInputError, SyncError, LLMError, RecreateError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
