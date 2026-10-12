"""Manifest resolution and reference-annotation loading (eval-only)."""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff", ".opus"}


@dataclass
class Clip:
    id: str
    desc: str = ""
    audio: str | None = None
    beats: np.ndarray | None = None
    downbeats: np.ndarray | None = None
    midi: str | None = None
    steady_tempo: bool | None = None
    generated: bool = False
    truth: object | None = None
    part3: bool = False
    ref_files: dict = field(default_factory=dict)


def first_match(patterns) -> str | None:
    for pat in patterns or []:
        hits = sorted(h for h in glob.glob(pat, recursive=True) if Path(h).is_file())
        if hits:
            return hits[0]
    return None


def first_audio(patterns) -> str | None:
    for pat in patterns or []:
        hits = sorted(h for h in glob.glob(pat, recursive=True)
                      if Path(h).is_file() and Path(h).suffix.lower() in AUDIO_EXT)
        if hits:
            return hits[0]
    return None


def load_times(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Beat annotation file -> (beat times, downbeat times or None).

    Supports: one time per line; "time position" per line (position 1 = downbeat,
    as in .beats files); JSON list or {"beats": [...], "downbeats": [...]}; JAMS.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() in (".json", ".jams"):
        d = json.loads(text)
        if isinstance(d, list):
            return np.array(sorted(float(x) for x in d)), None
        if "annotations" in d:  # JAMS
            for ann in d["annotations"]:
                if ann.get("namespace", "").startswith("beat"):
                    t = np.array([o["time"] for o in ann["data"]])
                    pos = [o.get("value") for o in ann["data"]]
                    db = np.array([o["time"] for o in ann["data"] if o.get("value") in (1, 1.0)]) if any(pos) else None
                    return t, db
        return np.array(d.get("beats", [])), (np.array(d["downbeats"]) if "downbeats" in d else None)
    rows = []
    for line in text.splitlines():
        line = line.strip().replace(",", " ").replace("\t", " ")
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            rows.append([float(x) for x in parts[:2]])
        except ValueError:
            continue
    if not rows:
        return np.zeros(0), None
    if all(len(r) >= 2 for r in rows):
        arr = np.array([r[:2] for r in rows])
        pos = arr[:, 1]
        if np.allclose(pos, np.round(pos)) and pos.min() >= 1:
            return arr[:, 0], arr[pos == 1, 0]
    return np.array([r[0] for r in rows]), None


def load_manifest(path="evals/manifest.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def resolve_clips(manifest: dict, gen_dir: Path | None = None) -> list[Clip]:
    out = []
    for c in manifest.get("clips", []):
        clip = Clip(id=c["id"], desc=c.get("desc", ""), audio=first_audio(c.get("audio")),
                    steady_tempo=c.get("steady_tempo"), part3=bool(c.get("part3")))
        bf = first_match(c.get("beats"))
        if bf:
            clip.beats, db = load_times(bf)
            clip.ref_files["beats"] = bf
            if db is not None and len(db):
                clip.downbeats = db
        dbf = first_match(c.get("downbeats"))
        if dbf and dbf != bf:
            clip.downbeats, _ = load_times(dbf)
            clip.ref_files["downbeats"] = dbf
        mf = first_match(c.get("midi"))
        if mf:
            clip.midi = mf
            clip.ref_files["midi"] = mf
        out.append(clip)
    if gen_dir is not None:
        out += render_generated(manifest, gen_dir)
    known = {Path(c.audio).resolve() for c in out if c.audio}
    for d in manifest.get("extra_audio_dirs", []):
        for f in sorted(Path(d).glob("*")):
            if f.suffix.lower() in AUDIO_EXT and f.resolve() not in known:
                out.append(Clip(id=f"user:{f.stem}", desc="user-supplied audio (no reference)", audio=str(f)))
    return out


def render_generated(manifest: dict, gen_dir: Path) -> list[Clip]:
    from evals import synth
    gen_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for g in manifest.get("generated", []):
        kw = dict(g["synth"])
        kw["segments"] = tuple(tuple(s) for s in kw["segments"])
        sr = kw.get("sr", 44100)
        audio, truth = synth.make_track(**kw)
        p = gen_dir / f"{g['id']}.wav"
        synth.write(p, audio, sr, g.get("subtype", "PCM_16"))
        out.append(Clip(id=g["id"], desc=g.get("desc", ""), audio=str(p), beats=truth.beats,
                        downbeats=truth.downbeats, steady_tempo=g.get("steady_tempo"), generated=True, truth=truth))
    return out
