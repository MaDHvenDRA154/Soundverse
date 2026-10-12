"""Eval entry point:  python -m evals.run {part1,part2,part3,robustness,all} [--manifest PATH]"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.run")
    ap.add_argument("what", choices=["part1", "part2", "part3", "robustness", "all", "demo"])
    ap.add_argument("--manifest", default="evals/manifest.yaml")
    ap.add_argument("--slow", action="store_true", help="robustness: include the 12-minute clip")
    ap.add_argument("--live", action="store_true",
                    help="bypass the LLM answer cache (same as LLM_CACHE=off): every Gemini call goes to the API")
    a = ap.parse_args(argv)
    if a.live:
        import os
        os.environ["LLM_CACHE"] = "off"
    todo = ["robustness", "part1", "part2", "part3"] if a.what in ("all", "demo") else [a.what]
    for w in todo:
        print(f"=== {w} ===", flush=True)
        if w == "robustness":
            from evals import robustness
            robustness.run(include_slow=a.slow or a.what == "all")
        elif w == "part1":
            from evals import part1
            part1.run(a.manifest)
        elif w == "part2":
            try:
                from evals import part2
            except ImportError:
                print("part2 harness not built yet", file=sys.stderr)
                continue
            part2.run(a.manifest)
        elif w == "part3":
            try:
                from evals import part3
            except ImportError:
                print("part3 harness not built yet", file=sys.stderr)
                continue
            part3.run(a.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
