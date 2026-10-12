"""Click audition: the original audio plus a click on every beat (accented on
downbeats). Streamed block by block; same rate, channels and bit depth as
the source."""
from __future__ import annotations

import numpy as np

from beat_agent import config
from beat_agent.analysis.schema import BeatGrid
from beat_agent.io.audio import SourceInfo, iter_blocks, open_writer


def click_wave(sr: int, accent: bool) -> np.ndarray:
    n = int(config.CLICK_DURATION_S * sr)
    t = np.arange(n) / sr
    f = config.CLICK_ACCENT_FREQ_HZ if accent else config.CLICK_FREQ_HZ
    g = config.CLICK_ACCENT_GAIN if accent else config.CLICK_GAIN
    return (g * np.sin(2 * np.pi * f * t) * np.exp(-t / config.CLICK_DECAY_S)).astype(np.float32)


def write_click_audition(src: SourceInfo, grid: BeatGrid, out_path) -> dict:
    sr = src.samplerate
    clicks = {False: click_wave(sr, False), True: click_wave(sr, True)}
    down = set(np.round(np.array(grid.downbeats) * sr).astype(np.int64).tolist())
    events = [(int(round(b.time * sr)), int(round(b.time * sr)) in down) for b in grid.beats]
    events.sort()
    ei = 0
    pos = 0
    with open_writer(out_path, src) as w:
        for blk in iter_blocks(src):
            n = len(blk)
            out = blk * config.AUDITION_MUSIC_GAIN
            # rewind to any click that started before this block but still rings into it
            while ei > 0 and events[ei - 1][0] + len(clicks[True]) > pos:
                ei -= 1
            while ei < len(events) and events[ei][0] < pos + n:
                start, acc = events[ei]
                c = clicks[acc]
                a, b = max(start, pos), min(start + len(c), pos + n)
                if b > a:
                    out[a - pos:b - pos] += c[a - start:b - start, None]
                ei += 1
            w.write(out)
            pos += n
        clipped = w.clips
    return {"path": str(out_path), "n_clicks": len(events), "n_accents": sum(1 for e in events if e[1]),
            "clipped_samples": clipped}
