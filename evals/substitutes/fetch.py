"""Fetch the substitute real-music clips (GTZAN excerpts + public beat annotations).

SUBSTITUTES, NOT THE TEST PACK. Used only while the official test pack is
missing, reported in their own table, and never used to choose a parameter.

Why GTZAN: the Beat This! checkpoint we use ("final0") was trained on all of its
datasets *except* GTZAN, so these clips are held out for the tracker (Ballroom,
Hainsworth etc. were in its training data and would flatter it).

Selection rule, fixed before any clip was scored: track 00000 of each genre that
best matches a test-pack material type -- disco (steady four-on-the-floor, like
A1/A7), hiphop (A2), rock (live band, A3), jazz (live, swung), country (steady pop).

Audio: 30 s clips from the GTZAN tarball on the Hugging Face hub (streamed; we stop
reading once the five members are found). Annotations: beats + downbeats from
github.com/TempoBeatDownbeat/gtzan_tempo_beat. Neither states a licence, so
neither is committed: both land in evals/substitutes/{audio,annotations}/
(gitignored). Run:  uv run python -m evals.substitutes.fetch
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIO = HERE / "audio"
ANN = HERE / "annotations"
TARBALL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
ANN_URL = "https://raw.githubusercontent.com/TempoBeatDownbeat/gtzan_tempo_beat/main/beats/gtzan_{g}_{n}.beats"
PICKS = [("disco", "00000"), ("hiphop", "00000"), ("rock", "00000"), ("jazz", "00000"), ("country", "00000")]


def main() -> int:
    AUDIO.mkdir(parents=True, exist_ok=True)
    ANN.mkdir(parents=True, exist_ok=True)
    for g, n in PICKS:
        dst = ANN / f"gtzan_{g}_{n}.beats"
        if not dst.exists():
            urllib.request.urlretrieve(ANN_URL.format(g=g, n=n), dst)
            print("annotation", dst.name)
    want = {f"{g}.{n}.wav" for g, n in PICKS if not (AUDIO / f"gtzan_{g}_{n}.wav").exists()}
    if not want:
        print("audio already present")
        return 0
    print(f"streaming {TARBALL} for {len(want)} clips (up to 1.2 GB)...", flush=True)
    with urllib.request.urlopen(TARBALL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            name = Path(m.name).name
            if name in want:
                g, n, _ = name.split(".")
                data = tar.extractfile(m).read()
                (AUDIO / f"gtzan_{g}_{n}.wav").write_bytes(data)
                want.discard(name)
                print("audio", name, flush=True)
                if not want:
                    break
    if want:
        print("missing from tarball:", want, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
