"""Pitch-preserving time-stretch through the Rubber Band CLI (R3 engine)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class StretchError(RuntimeError):
    pass


def rubberband_exe() -> str:
    env = os.environ.get("BEAT_AGENT_RUBBERBAND")
    if env and Path(env).exists():
        return env
    exe = shutil.which("rubberband")
    if exe:
        return exe
    local = Path(__file__).resolve().parents[2] / ".tools" / "bin" / ("rubberband.exe" if os.name == "nt" else "rubberband")
    if local.exists():
        return str(local)
    raise StretchError("Rubber Band CLI not found. Install `rubberband-cli` (apt/brew) or set BEAT_AGENT_RUBBERBAND.")


def timemap_stretch(y: np.ndarray, sr: int, time_map: list[tuple[int, int]], out_len: int) -> np.ndarray:
    """Stretch y (n, ch) so that source frame s_i lands on target frame t_i.

    time_map must be increasing in both columns. Output is trimmed/padded to
    exactly out_len frames.
    """
    if y.ndim == 1:
        y = y[:, None]
    ratio = out_len / len(y)
    with tempfile.TemporaryDirectory(prefix="ba_rb_") as td:
        src, dst, mp = Path(td, "in.wav"), Path(td, "out.wav"), Path(td, "map.txt")
        sf.write(str(src), y, sr, subtype="FLOAT")
        # the map must not repeat the origin; rubberband anchors 0 -> 0 itself
        lines = [f"{int(s)} {int(t)}" for s, t in time_map if s > 0]
        mp.write_text("\n".join(lines) + "\n")
        cmd = [rubberband_exe(), "-q", "--fine", "--timemap", str(mp), "-t", f"{ratio:.10f}", str(src), str(dst)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not dst.exists():
            raise StretchError(f"rubberband failed: {res.stderr.strip()[:400]}")
        out, sr2 = sf.read(str(dst), dtype="float32", always_2d=True)
    if len(out) >= out_len:
        out = out[:out_len]
    else:
        out = np.vstack([out, np.zeros((out_len - len(out), out.shape[1]), np.float32)])
    return out
