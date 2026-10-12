"""Audio input/output for arbitrary user files.

Every component reads audio through `open_source`, which gives a seekable PCM
file plus the format facts exports must preserve (rate, channels, subtype).
Reads are streamed in blocks so a 12-minute 96 kHz file never has to sit in RAM
as one array.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import soxr

from beat_agent import config


class AudioInputError(ValueError):
    """The file cannot be used as audio input. Message is user-facing."""


@dataclass(frozen=True)
class SourceInfo:
    path: str              # what the user gave us
    readable_path: str     # seekable PCM file we actually read (== path for WAV/FLAC)
    samplerate: int
    channels: int
    frames: int
    input_format: str      # e.g. WAV, FLAC, MP3
    input_subtype: str     # e.g. PCM_16, PCM_24, FLOAT, MPEG_LAYER_III
    export_subtype: str    # subtype every export of this source is written with
    lossy: bool

    @property
    def duration(self) -> float:
        return self.frames / self.samplerate

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_s"] = round(self.duration, 6)
        return d


_PCM_SUBTYPES = {"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}


def cache_dir() -> Path:
    d = Path(os.environ.get("BEAT_AGENT_CACHE", ".cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _file_digest(path: Path) -> str:
    h = hashlib.sha1()
    st = path.stat()
    h.update(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _decode_to_wav(path: Path) -> Path:
    """Decode a non-native format once to a float WAV in the cache."""
    out = cache_dir() / "decoded" / f"{path.stem}-{_file_digest(path)}.wav"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.wav")
    exe = ffmpeg_exe()
    if exe:
        # -vn drops cover art streams; f32le keeps the decoder's full precision.
        cmd = [exe, "-v", "error", "-y", "-i", str(path), "-vn", "-map", "0:a:0",
               "-c:a", "pcm_f32le", str(tmp)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not tmp.exists():
            raise AudioInputError(f"Could not decode {path.name}: {res.stderr.strip()[:300] or 'ffmpeg failed'}")
    else:
        # libsndfile can read MP3/OGG itself; stream it across.
        try:
            with sf.SoundFile(str(path)) as src, sf.SoundFile(str(tmp), "w", src.samplerate, src.channels, "FLOAT") as dst:
                for block in src.blocks(blocksize=config.IO_BLOCK_FRAMES, dtype="float32", always_2d=True):
                    dst.write(block)
        except Exception as e:  # pragma: no cover - only without ffmpeg
            raise AudioInputError(f"Could not decode {path.name} and ffmpeg is not available: {e}") from e
    tmp.replace(out)
    return out


def open_source(path: str | os.PathLike) -> SourceInfo:
    p = Path(path)
    if not p.exists():
        raise AudioInputError(f"File not found: {p}")
    if p.stat().st_size == 0:
        raise AudioInputError(f"{p.name} is empty (0 bytes).")
    native = None
    try:
        native = sf.info(str(p))
    except Exception:
        native = None
    if native is not None and native.format in config.NATIVE_FORMATS and native.subtype in _PCM_SUBTYPES:
        readable, info = p, native
        fmt, subtype, lossy = native.format, native.subtype, False
        export_subtype = native.subtype
    else:
        readable = _decode_to_wav(p)
        info = sf.info(str(readable))
        fmt = native.format if native is not None else p.suffix.lstrip(".").upper()
        subtype = native.subtype if native is not None else "LOSSY/UNKNOWN"
        lossy = True
        export_subtype = config.LOSSY_EXPORT_SUBTYPE
    if info.frames == 0:
        raise AudioInputError(f"{p.name} decodes to zero samples.")
    return SourceInfo(
        path=str(p), readable_path=str(readable), samplerate=int(info.samplerate),
        channels=int(info.channels), frames=int(info.frames), input_format=fmt,
        input_subtype=subtype, export_subtype=export_subtype, lossy=lossy,
    )


def iter_blocks(src: SourceInfo, start: int = 0, stop: int | None = None,
                block: int = config.IO_BLOCK_FRAMES) -> Iterator[np.ndarray]:
    """Yield float32 blocks of shape (n, channels) covering [start, stop)."""
    stop = src.frames if stop is None else min(stop, src.frames)
    with sf.SoundFile(src.readable_path) as f:
        f.seek(max(0, start))
        pos = max(0, start)
        while pos < stop:
            n = min(block, stop - pos)
            data = f.read(n, dtype="float32", always_2d=True)
            if len(data) == 0:
                break
            pos += len(data)
            yield data


def read_range(src: SourceInfo, start: int, stop: int) -> np.ndarray:
    """Read frames [start, stop), zero-padded where the range leaves the file."""
    n = max(0, stop - start)
    out = np.zeros((n, src.channels), dtype=np.float32)
    a, b = max(start, 0), min(stop, src.frames)
    if b > a:
        with sf.SoundFile(src.readable_path) as f:
            f.seek(a)
            data = f.read(b - a, dtype="float32", always_2d=True)
        out[a - start:a - start + len(data)] = data
    return out


def load_analysis_mono(src: SourceInfo, sr: int = config.ANALYSIS_SR) -> np.ndarray:
    """Mono mixdown resampled to `sr`, streamed so memory stays bounded."""
    pieces = []
    if src.samplerate == sr:
        for blk in iter_blocks(src):
            pieces.append(blk.mean(axis=1))
    else:
        rs = soxr.ResampleStream(src.samplerate, sr, 1, dtype="float32", quality="HQ")
        blocks = iter_blocks(src)
        prev = next(blocks, None)
        while prev is not None:
            nxt = next(blocks, None)
            pieces.append(rs.resample_chunk(np.ascontiguousarray(prev.mean(axis=1)), last=nxt is None))
            prev = nxt
    y = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return y.astype(np.float32, copy=False)


class ExportWriter:
    """WAV writer with the source's rate, channel count and bit depth.

    Integer PCM subtypes are clipped explicitly: libsndfile's float->int
    conversion wraps on overflow, which would turn a 0.1 dB overshoot after a
    gain change into a full-scale click.
    """

    def __init__(self, path, samplerate: int, channels: int, subtype: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.subtype = subtype
        self.clips = 0
        self._f = sf.SoundFile(str(path), "w", samplerate, channels, subtype, format="WAV")

    def write(self, data: np.ndarray) -> None:
        if self.subtype not in ("FLOAT", "DOUBLE"):
            over = np.abs(data) > 1.0
            if over.any():
                self.clips += int(over.sum())
                data = np.clip(data, -1.0, 1.0)
        self._f.write(data)

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_writer(path: str | os.PathLike, src: SourceInfo, channels: int | None = None,
                samplerate: int | None = None) -> ExportWriter:
    return ExportWriter(path, samplerate or src.samplerate, channels or src.channels, src.export_subtype)


def write_like(path: str | os.PathLike, data: np.ndarray, src: SourceInfo) -> None:
    with open_writer(path, src, channels=data.shape[1] if data.ndim == 2 else 1) as w:
        w.write(data)
