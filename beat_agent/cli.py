"""beat-agent command line.

  beat-agent analyze <file> [--out DIR]
  beat-agent sync <master> <other> --at-bar N [--b-bar M] [--out DIR]
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
    a = ap.parse_args(argv)
    from beat_agent.io.audio import AudioInputError
    from beat_agent.sync.sync import SyncError
    try:
        return a.fn(a)
    except (AudioInputError, SyncError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
