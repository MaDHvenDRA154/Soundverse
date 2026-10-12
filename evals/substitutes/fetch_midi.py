"""Build SUBSTITUTE Part 3 clips (P1-P4 are missing) from public MIDI.

NOT the test pack. Each clip is a MIDI excerpt rendered to audio with a General
MIDI soundfont; the excerpt MIDI is the reference (except the P4 stand-in,
which is reported without a reference, as in the brief). Selection rules are
fixed here, before any transcription was scored, and never revisited:

  subP1  2-bar drum loop      Groove MIDI Dataset (Magenta, CC BY 4.0): the first
                              4/4 "beat" in info.csv order of >= 8 bars; bars 2-3.
  subP2  4-bar drums + bass   Lakh MIDI "clean" set (CC BY 4.0): the first file in
                              archive order with one tempo, 4/4, a drum track, a GM
                              bass (programs 32-39) and >= 16 bars; bars 9-12, drums
                              and bass only.
  subP3  8-bar piano melody   POP909 (MIT) song 001: MELODY + PIANO tracks (both
         with chords          rendered as acoustic piano); 8 bars from bar 9.
  subP4  8-bar dense mix      Lakh: the next qualifying file after subP2 with drums
         (no reference used)  and >= 6 other instruments; bars 17-24, all tracks.

Rendering: FluidSynth, reverb/chorus off, 44.1 kHz stereo, soundfont TimGM6mb
(GPL-2, Debian package `timgm6mb-soundfont`). Downloads land in
evals/substitutes/part3/ (gitignored). Run: uv run python -m evals.substitutes.fetch_midi
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pretty_midi

HERE = Path(__file__).resolve().parent
OUT = HERE / "part3"
DL = OUT / "_downloads"
GMD = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
LAKH = "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"
POP909 = "https://raw.githubusercontent.com/music-x-lab/POP909-Dataset/master/POP909/001/001.mid"
SF2_DEB = "http://deb.debian.org/debian/pool/main/t/timgm6mb-soundfont/timgm6mb-soundfont_1.3-5_all.deb"
FS_WIN = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.4.4/fluidsynth-2.4.4-win10-x64.zip"
# the Windows FluidSynth build links SDL3.dll but does not ship it
SDL3_WIN = "https://github.com/libsdl-org/SDL/releases/download/release-3.4.18/SDL3-3.4.18-win32-x64.zip"


def _get(url: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("download", url, flush=True)
        urllib.request.urlretrieve(url, dst)
    return dst


# ------------------------------------------------------------------ helpers
def bar_times(pm: pretty_midi.PrettyMIDI) -> list[float]:
    return list(pm.get_downbeats())


def excerpt(pm: pretty_midi.PrettyMIDI, bar0: int, nbars: int, keep=None) -> pretty_midi.PrettyMIDI:
    """Bars bar0 .. bar0+nbars-1 (1-based), shifted to start at 0, constant tempo."""
    db = bar_times(pm)
    if len(db) < bar0 + nbars:
        raise ValueError("too few bars")
    t0, t1 = db[bar0 - 1], db[bar0 - 1 + nbars]
    tempo = pm.get_tempo_changes()[1][0]
    out = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    out.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
    for inst in pm.instruments:
        if keep is not None and not keep(inst):
            continue
        ni = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            if t0 <= n.start < t1 - 1e-6:
                ni.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.start - t0, min(n.end, t1) - t0))
        if ni.notes:
            out.instruments.append(ni)
    out._meta = {"source_seconds": [t0, t1], "tempo": float(tempo), "bars": nbars}  # noqa: SLF001
    return out


def single_tempo_44(pm) -> bool:
    _, tempi = pm.get_tempo_changes()
    ts = pm.time_signature_changes
    return len(tempi) == 1 and all(t.numerator == 4 and t.denominator == 4 for t in ts) and len(ts) >= 1


# ------------------------------------------------------------------ sources
def pick_gmd() -> pretty_midi.PrettyMIDI:
    z = zipfile.ZipFile(_get(GMD, DL / "groove-midionly.zip"))
    info = next(n for n in z.namelist() if n.endswith("info.csv"))
    root = info.rsplit("/", 1)[0]
    for row in csv.DictReader(io.TextIOWrapper(z.open(info), encoding="utf-8")):
        if row["beat_type"] != "beat" or row["time_signature"] != "4-4":
            continue
        pm = pretty_midi.PrettyMIDI(io.BytesIO(z.read(f"{root}/{row['midi_filename']}")))
        if len(bar_times(pm)) >= 9:
            print("subP1 from GMD", row["midi_filename"], flush=True)
            ex = excerpt(pm, 2, 2)
            ex._meta["source"] = f"Groove MIDI Dataset {row['midi_filename']} (CC BY 4.0)"
            return ex
    raise RuntimeError("no GMD file matched")


def pick_lakh() -> tuple:
    p2 = p4 = None
    print("streaming Lakh clean_midi (stops at the first matches)...", flush=True)
    with urllib.request.urlopen(LAKH) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile() or not m.name.lower().endswith(".mid"):
                continue
            try:
                pm = pretty_midi.PrettyMIDI(io.BytesIO(tar.extractfile(m).read()))
            except Exception:
                continue
            if not single_tempo_44(pm) or len(bar_times(pm)) < 25:
                continue
            drums = [i for i in pm.instruments if i.is_drum and i.notes]
            bass = [i for i in pm.instruments if not i.is_drum and 32 <= i.program <= 39 and i.notes]
            others = [i for i in pm.instruments if not i.is_drum and i.notes]
            if p2 is None and drums and bass:
                ex = excerpt(pm, 9, 4, keep=lambda i: i.is_drum or 32 <= i.program <= 39)
                if any(i.is_drum for i in ex.instruments) and any(not i.is_drum for i in ex.instruments):
                    ex._meta["source"] = f"Lakh MIDI clean {m.name} (CC BY 4.0)"
                    p2 = ex
                    print("subP2 from", m.name, flush=True)
                    continue
            if p2 is not None and p4 is None and drums and len(others) >= 6:
                ex = excerpt(pm, 17, 8)
                ex._meta["source"] = f"Lakh MIDI clean {m.name} (CC BY 4.0)"
                p4 = ex
                print("subP4 from", m.name, flush=True)
            if p2 is not None and p4 is not None:
                break
    if p2 is None or p4 is None:
        raise RuntimeError("Lakh: no matching files")
    return p2, p4


def pick_pop909() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(str(_get(POP909, DL / "pop909_001.mid")))
    for inst in pm.instruments:          # melody and accompaniment both as acoustic grand piano
        inst.program = 0
    ex = excerpt(pm, 9, 8, keep=lambda i: i.name.upper() in ("MELODY", "PIANO"))
    ex._meta["source"] = "POP909 song 001, MELODY + PIANO tracks (MIT)"
    return ex


# ------------------------------------------------------------------ render
def soundfont() -> Path:
    for p in ("/usr/share/sounds/sf2/TimGM6mb.sf2", "/usr/share/soundfonts/TimGM6mb.sf2"):
        if Path(p).exists():
            return Path(p)
    sf2 = DL / "TimGM6mb.sf2"
    if sf2.exists():
        return sf2
    deb = _get(SF2_DEB, DL / "timgm6mb.deb")
    data = _ar_member(deb.read_bytes(), "data.tar")
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        m = next(x for x in t.getmembers() if x.name.endswith(".sf2"))
        sf2.write_bytes(t.extractfile(m).read())
    return sf2


def _ar_member(blob: bytes, prefix: str) -> bytes:
    """Minimal `ar` reader for .deb packages."""
    pos = 8
    while pos < len(blob):
        name = blob[pos:pos + 16].decode().strip().rstrip("/")
        size = int(blob[pos + 48:pos + 58].decode().strip())
        body = blob[pos + 60:pos + 60 + size]
        if name.startswith(prefix):
            return body
        pos += 60 + size + (size % 2)
    raise RuntimeError(f"{prefix} not in .deb")


def fluidsynth() -> str:
    exe = shutil.which("fluidsynth")
    if exe:
        return exe
    if os.name == "nt":
        d = HERE.parents[1] / ".tools" / "fluidsynth"
        cand = list(d.rglob("fluidsynth.exe"))
        if not cand:
            z = zipfile.ZipFile(_get(FS_WIN, DL / "fluidsynth-win.zip"))
            z.extractall(d)
            cand = list(d.rglob("fluidsynth.exe"))
        bindir = cand[0].parent
        if not (bindir / "SDL3.dll").exists():
            z = zipfile.ZipFile(_get(SDL3_WIN, DL / "sdl3-win.zip"))
            (bindir / "SDL3.dll").write_bytes(z.read(next(n for n in z.namelist() if n.endswith("SDL3.dll"))))
        return str(cand[0])
    raise RuntimeError("fluidsynth not found (apt install fluidsynth)")


def render(mid: Path, wav: Path, length_s: float) -> None:
    """FluidSynth render, trimmed to exactly the excerpt's bars (FluidSynth keeps
    rendering release tails, e.g. 12 s of cymbal after a 3.5 s loop), with a
    10 ms fade so the cut is not a click."""
    import numpy as np
    import soundfile as sf
    cmd = [fluidsynth(), "-ni", "-q", "-o", "synth.reverb.active=0", "-o", "synth.chorus.active=0",
           "-r", "44100", "-F", str(wav), str(soundfont()), str(mid)]
    subprocess.run(cmd, check=True, capture_output=True)
    y, sr = sf.read(str(wav), always_2d=True)
    n = int(round(length_s * sr))
    y = y[:n] if len(y) >= n else np.vstack([y, np.zeros((n - len(y), y.shape[1]))])
    f = int(0.010 * sr)
    y[-f:] *= np.linspace(1, 0, f)[:, None]
    sf.write(str(wav), y, sr, subtype="PCM_16")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = {"subP1": pick_gmd(), "subP3": pick_pop909()}
    clips["subP2"], clips["subP4"] = pick_lakh()
    meta = {}
    for cid, pm in clips.items():
        mid, wav = OUT / f"{cid}.mid", OUT / f"{cid}.wav"
        pm.write(str(mid))
        render(mid, wav, pm._meta["source_seconds"][1] - pm._meta["source_seconds"][0])
        meta[cid] = dict(pm._meta, midi=str(mid.relative_to(HERE.parents[1])), audio=str(wav.relative_to(HERE.parents[1])),
                         reference_used=cid != "subP4",
                         instruments=[f"{'drums' if i.is_drum else pretty_midi.program_to_instrument_name(i.program)} "
                                      f"({len(i.notes)} notes)" for i in pm.instruments])
        print(cid, meta[cid]["instruments"], flush=True)
    (OUT / "clips.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
