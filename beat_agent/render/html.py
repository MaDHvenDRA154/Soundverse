"""Score JSON -> one self-contained HTML page (deterministic; no LLM involved).

The page is a fixed template; the only thing that varies is the score data,
inlined as JSON. Same score in, same bytes out.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from beat_agent.transcribe.score import Score

TEMPLATE = Path(__file__).with_name("template.html")
PLACEHOLDER = "/*SCORE_JSON*/"


def render_html(score: Score) -> str:
    data = json.dumps(score.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
    # inline JSON inside <script>: neutralise anything that could close the tag
    data = data.replace("</", "<\\/").replace("<!--", "<\\!--")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.count(PLACEHOLDER) == 1
    return html.replace(PLACEHOLDER, data)


def write_html(score: Score, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_html(score), encoding="utf-8")
    return str(path)


# Things a recreation page must never contain (checked by tests and evals).
FORBIDDEN = [
    (r"https?:", "a URL"),
    (r"\bfetch\s*\(", "fetch()"),
    (r"XMLHttpRequest|WebSocket|EventSource|navigator\.sendBeacon", "a network API"),
    (r"\bimport\s*[\(\{\"'\w]", "a JS import"),
    (r"<audio\b|<video\b|<source\b", "a media element"),
    (r"data:audio|data:application/octet", "embedded audio"),
    (r";base64,", "a base64 data URI"),
    (r"<link\b|<iframe\b|<img\b", "an external resource tag"),
    (r"\bsrc\s*=", "a src attribute"),
]


def safety_violations(html: str) -> list[str]:
    return [f"contains {what} ({pat})" for pat, what in FORBIDDEN if re.search(pat, html, flags=re.I)]
