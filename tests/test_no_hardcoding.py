"""Mechanical guard: the analysis/edit/agent code must not know about the test pack.

The reviewers re-run on held-out audio. Anything here that mentions a test-pack
clip, reads reference annotations or MIDI, or imports the eval harness is a
leak and fails the build.
"""
from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "beat_agent"

# Test-pack clip IDs (A1..A9, P1..P9) as standalone tokens. Note names such as
# "A4" would also match: the package expresses pitch as MIDI numbers instead.
CLIP_ID = re.compile(r"(?<![A-Za-z0-9_])[AP][1-9](?![A-Za-z0-9_])")

# Regexes over lower-cased source, written so "preference" or
# "from __future__ import annotations" do not trip them.
FORBIDDEN_PATTERNS = [
    r"testpack", r"test_pack",
    r"annotation",                 # reference beat/downbeat files (also read_annotations)
    r"manifest",                   # the eval manifest maps clips to references
    r"(?<!p)reference",            # no "calibrate against reference"; "preference" is fine
    r"\.midi?\b",                  # MIDI files are read only by evals/
    r"pretty_midi",
    r"(import|from)\s+mir_eval",   # scoring belongs to evals/
    r"(import|from)\s+evals\b",
]
FUTURE_IMPORT = "from __future__ import annotations"


def _sources():
    files = sorted(PKG.rglob("*.py")) + sorted(PKG.rglob("*.yaml")) + sorted(PKG.rglob("*.html")) + sorted(PKG.rglob("*.js"))
    assert files, "no sources found -- wrong path?"
    return files


def test_no_clip_ids_in_package():
    hits = []
    for f in _sources():
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if CLIP_ID.search(line):
                hits.append(f"{f.relative_to(PKG.parent)}:{n}: {line.strip()}")
    assert not hits, "test-pack clip IDs in package code:\n" + "\n".join(hits)


def test_no_reference_or_eval_access_in_package():
    hits = []
    for f in _sources():
        text = f.read_text(encoding="utf-8").lower()
        for n, line in enumerate(text.splitlines(), 1):
            line = line.replace(FUTURE_IMPORT, "")
            for pat in FORBIDDEN_PATTERNS:
                if re.search(pat, line):
                    hits.append(f"{f.relative_to(PKG.parent)}:{n}: /{pat}/ in: {line.strip()}")
    assert not hits, "package code touches references / eval harness:\n" + "\n".join(hits)


def test_guard_actually_detects_leaks(tmp_path):
    """The regexes above must fire on the patterns they are meant to catch."""
    bad = 'if "A6" in path: meter = (6, 8)\nref = load("testpack/A1/annotations.beats")\n'
    assert CLIP_ID.search(bad.splitlines()[0])
    assert any(re.search(p, bad.lower()) for p in FORBIDDEN_PATTERNS)
    for leak in ["import mir_eval", "pm = pretty_midi.PrettyMIDI(x)", "load('clip.mid')",
                 "beats = read_annotations(p)", "from evals.run import x"]:
        assert any(re.search(p, leak.lower()) for p in FORBIDDEN_PATTERNS), leak
    for ok in [FUTURE_IMPORT, "# a mild preference for 4/4", "middle = 3"]:
        assert not any(re.search(p, ok.replace(FUTURE_IMPORT, "").lower()) for p in FORBIDDEN_PATTERNS), ok
    assert not CLIP_ID.search("ANALYSIS_SR = 22050  # a1b2 x_A1 A10")
