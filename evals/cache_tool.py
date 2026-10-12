"""The committed LLM answer cache (evals/llm_cache/): secret check and pruning.

    python -m evals.cache_tool check   # every entry has only the allowed fields and nothing key-like
    python -m evals.cache_tool prune   # keep only the entries the committed results replay

Each entry is one model answer: {"text", "usage", "model", "fetched_at"}. The cache key
(the file name) is a sha256 of model + system prompt + user contents + schema, so neither
the prompt nor any request metadata is stored.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ALLOWED = {"text", "usage", "model", "fetched_at"}
USAGE = {"prompt_tokens", "output_tokens", "thinking_tokens"}
SECRET = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|authorization|x-goog-api-key|bearer\s|api[_-]?key|"
                    r"password|secret|token=|-----BEGIN", re.I)
RESULTS = Path("evals/results")


def check(d: Path | None = None) -> list[str]:
    from beat_agent.agent.llm import _cache_dir
    d = d or _cache_dir()
    problems = []
    for f in sorted(d.glob("*")):
        if f.suffix != ".json" or not re.fullmatch(r"[0-9a-f]{64}", f.stem):
            problems.append(f"{f.name}: not a cache entry")
            continue
        raw = f.read_text(encoding="utf-8")
        e = json.loads(raw)
        if set(e) != ALLOWED:
            problems.append(f"{f.name}: fields {sorted(e)}")
        if not set(e.get("usage", {})) <= USAGE:
            problems.append(f"{f.name}: usage fields {sorted(e['usage'])}")
        if not e.get("fetched_at"):
            problems.append(f"{f.name}: no fetched_at")
        for m in SECRET.findall(raw):
            problems.append(f"{f.name}: secret-like string {m!r}")
    return problems


def referenced_keys() -> set[str]:
    keys = set()
    p2, p3 = RESULTS / "part2.json", RESULTS / "part3.json"
    if p2.exists():
        keys |= {a["cache_key"] for s in json.loads(p2.read_text()).get("sessions", [])
                 for t in s.get("turns", []) for a in t.get("llm_attempts", []) if a.get("cache_key")}
    if p3.exists():
        keys |= {c["cache_key"] for t in (json.loads(p3.read_text()).get("chat") or {}).get("turns", [])
                 for c in t.get("llm_calls", []) if c.get("cache_key")}
    return keys


def prune() -> tuple[int, int]:
    from beat_agent.agent.llm import _cache_dir
    keep = referenced_keys()
    if not keep:
        raise SystemExit("no cache keys in evals/results (run the evals first); nothing pruned")
    files = list(_cache_dir().glob("*.json"))
    removed = [f for f in files if f.stem not in keep]
    missing = keep - {f.stem for f in files}
    if missing:
        raise SystemExit(f"{len(missing)} referenced answers are not in the cache; refusing to prune")
    for f in removed:
        f.unlink()
    return len(files) - len(removed), len(removed)


def main(argv=None) -> int:
    cmd = (argv or sys.argv[1:] or ["check"])[0]
    if cmd == "check":
        p = check()
        print("\n".join(p) if p else "cache OK: only answer text, token counts, model, fetch time")
        return 1 if p else 0
    if cmd == "prune":
        kept, removed = prune()
        print(f"kept {kept} entries referenced by evals/results, removed {removed}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
