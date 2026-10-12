"""Copy the committable Part 3 outputs into evals/samples/ (tracked in git).

Only substitutes whose manifest entry says `commit_outputs: true` are copied:
the input clip, its reference MIDI, and the recreation (score.json, page.html,
the offline render). Everything else under evals/out/ stays untracked and is
rebuilt by `make reproduce`. Run after `python -m evals.run part3`:
    python -m evals.samples
"""
from __future__ import annotations

import shutil
from pathlib import Path

from evals.references import first_audio, first_match, load_manifest

SAMPLES = Path("evals/samples")
OUT = Path("evals/out/part3")
RECREATION = ("score.json", "page.html", "render_1loop.wav")


def run(manifest_path="evals/manifest.yaml") -> list[str]:
    man = load_manifest(manifest_path)
    if SAMPLES.exists():
        shutil.rmtree(SAMPLES)   # a clip whose licence status changed must not linger
    done = []
    for c in man.get("part3_substitutes", []):
        if not c.get("commit_outputs"):
            continue
        audio, midi, rec = first_audio(c.get("audio")), first_match(c.get("midi")), OUT / c["id"]
        if not audio or not (rec / "score.json").exists():
            print(f"skip {c['id']}: run `python -m evals.run part3` first")
            continue
        d = SAMPLES / c["id"]
        d.mkdir(parents=True)
        shutil.copy2(audio, d / f"original{Path(audio).suffix}")
        if midi:
            shutil.copy2(midi, d / "reference.mid")
        for f in RECREATION:
            if (rec / f).exists():
                shutil.copy2(rec / f, d / f)
        (d / "ATTRIBUTION.md").write_text(
            f"# {c['id']} -- SUBSTITUTE (stands in for {c.get('stands_in_for')}; not the test pack)\n\n"
            f"{c.get('desc', '')}\n\n"
            f"- Source licence: {c.get('licence')}\n"
            f"- Cite: {c.get('attribution')}\n"
            "- `original.wav` is the excerpt MIDI rendered with FluidSynth and the TimGM6mb General MIDI soundfont\n"
            "  (GPL-2, Tim Brechbill; Debian package `timgm6mb-soundfont`).\n"
            "- `score.json`, `page.html`, `render_*.wav` are this repo's recreation of `original.wav`\n"
            "  (`python -m evals.run part3`); `reference.mid` is the excerpt used as the reference.\n",
            encoding="utf-8")
        done.append(c["id"])
        print(f"samples: {d}")
    return done


if __name__ == "__main__":
    run()
