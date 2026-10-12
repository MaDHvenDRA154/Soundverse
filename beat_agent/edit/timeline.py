"""Virtual edit timeline over the immutable source audio.

A Timeline is a tuple of Clips; each Clip plays source samples
[src_start, src_end) with a product of gain factors. All edit operations are
pure functions Timeline -> Timeline on *output* sample positions, so undo is
just keeping the previous value. Rendering streams from the source file.

Splices: wherever consecutive clips are not contiguous in the source, an
equal-power crossfade of SPLICE_CROSSFADE_S is centred on the splice point. It
uses source audio from just beyond each clip's edge (the "handles"), so the
output length is exactly the sum of the clip lengths and every grid time
stays exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from beat_agent import config
from beat_agent.io.audio import SourceInfo, iter_blocks, open_writer, read_range


@dataclass(frozen=True)
class Factor:
    kind: str            # "gain" | "fade_in" | "fade_out"
    shape: str = "equal_power"
    value: float = 1.0   # linear gain for kind == "gain"
    u0: float = 0.0      # fade phase at the clip's first sample
    u1: float = 1.0      # fade phase at the clip's end

    def curve(self, u: np.ndarray) -> np.ndarray:
        if self.kind == "gain":
            return np.full_like(u, self.value)
        u = np.clip(u, 0.0, 1.0)
        if self.shape == "linear":
            return u if self.kind == "fade_in" else 1.0 - u
        return np.sin(0.5 * np.pi * u) if self.kind == "fade_in" else np.cos(0.5 * np.pi * u)

    def split(self, frac: float) -> tuple["Factor", "Factor"]:
        um = self.u0 + (self.u1 - self.u0) * frac
        return replace(self, u1=um), replace(self, u0=um)


@dataclass(frozen=True)
class Clip:
    src_start: int
    src_end: int
    factors: tuple[Factor, ...] = field(default_factory=tuple)

    @property
    def length(self) -> int:
        return self.src_end - self.src_start

    def gains(self, rel: np.ndarray) -> np.ndarray:
        g = np.ones(len(rel))
        L = max(1, self.length)
        for f in self.factors:
            g = g * f.curve(f.u0 + (f.u1 - f.u0) * (rel / L))
        return g

    def split(self, at: int) -> tuple["Clip", "Clip"]:
        frac = at / self.length
        lf, rf = zip(*(f.split(frac) for f in self.factors)) if self.factors else ((), ())
        return (Clip(self.src_start, self.src_start + at, tuple(lf)),
                Clip(self.src_start + at, self.src_end, tuple(rf)))


@dataclass(frozen=True)
class Timeline:
    clips: tuple[Clip, ...]

    @staticmethod
    def whole(frames: int) -> "Timeline":
        return Timeline((Clip(0, frames),))

    @property
    def length(self) -> int:
        return sum(c.length for c in self.clips)

    def offsets(self) -> list[int]:
        out, o = [], 0
        for c in self.clips:
            out.append(o)
            o += c.length
        return out

    def splices(self) -> list[int]:
        """Output sample positions where the source is discontinuous."""
        offs = self.offsets()
        return [offs[i] for i in range(1, len(self.clips))
                if self.clips[i - 1].src_end != self.clips[i].src_start]

    def splice_points(self) -> list[tuple[int, int, int]]:
        """(output position, source end of the clip before, source start of the clip after)."""
        offs = self.offsets()
        return [(offs[i], self.clips[i - 1].src_end, self.clips[i].src_start) for i in range(1, len(self.clips))
                if self.clips[i - 1].src_end != self.clips[i].src_start]

    # ----------------------------------------------------------- primitives
    def split_at(self, pos: int) -> "Timeline":
        if pos <= 0 or pos >= self.length:
            return self
        out, o = [], 0
        for c in self.clips:
            if o < pos < o + c.length:
                a, b = c.split(pos - o)
                out += [a, b]
            else:
                out.append(c)
            o += c.length
        return Timeline(tuple(out))

    def _three(self, a: int, b: int) -> tuple[tuple[Clip, ...], tuple[Clip, ...], tuple[Clip, ...]]:
        t = self.split_at(a).split_at(b)
        left, mid, right, o = [], [], [], 0
        for c in t.clips:
            (left if o + c.length <= a else right if o >= b else mid).append(c)
            o += c.length
        return tuple(left), tuple(mid), tuple(right)

    # ------------------------------------------------------------- edits
    def remove(self, a: int, b: int) -> "Timeline":
        l, _, r = self._three(a, b)
        return Timeline(_merge(l + r))

    def keep_only(self, a: int, b: int) -> "Timeline":
        _, m, _ = self._three(a, b)
        return Timeline(_merge(m))

    def repeat(self, a: int, b: int, times_total: int) -> "Timeline":
        l, m, r = self._three(a, b)
        return Timeline(_merge(l + m * times_total + r))

    def insert_copy(self, a: int, b: int, at: int, copies: int = 1) -> "Timeline":
        _, m, _ = self._three(a, b)
        l2, _, r2 = self._three(at, at)
        return Timeline(_merge(l2 + m * copies + r2))

    def move(self, a: int, b: int, to: int) -> "Timeline":
        l, m, r = self._three(a, b)
        rest = Timeline(_merge(l + r))
        to2 = to if to <= a else to - (b - a)
        l2, _, r2 = rest._three(to2, to2)
        return Timeline(_merge(l2 + m + r2))

    def apply_factor(self, a: int, b: int, factor: Factor) -> "Timeline":
        """Apply a gain/fade over output [a, b); fade phase runs 0 -> 1 across it."""
        l, m, r = self._three(a, b)
        span = max(1, b - a)
        out, o = [], a
        for c in m:
            f = replace(factor, u0=(o - a) / span, u1=(o + c.length - a) / span) if factor.kind != "gain" else factor
            out.append(Clip(c.src_start, c.src_end, c.factors + (f,)))
            o += c.length
        return Timeline(_merge(l + tuple(out) + r))


def _merge(clips: tuple[Clip, ...]) -> tuple[Clip, ...]:
    """Re-join neighbours that are contiguous in the source with identical factors
    (keeps the clip list short after split-heavy edits; changes no sample)."""
    out: list[Clip] = []
    for c in clips:
        if c.length <= 0:
            continue
        if out and out[-1].src_end == c.src_start and not out[-1].factors and not c.factors:
            out[-1] = Clip(out[-1].src_start, c.src_end)
        else:
            out.append(c)
    return tuple(out)


# ------------------------------------------------------------------ render
def render(src: SourceInfo, tl: Timeline, out_path, crossfade_s: float = config.SPLICE_CROSSFADE_S) -> dict:
    sr = src.samplerate
    H = max(1, int(round(crossfade_s * sr / 2)))
    clips = tl.clips
    n = len(clips)

    def disc(i: int) -> bool:   # discontinuity between clip i and i+1
        return clips[i].src_end != clips[i + 1].src_start

    def xfade(i: int) -> np.ndarray:
        a, b = clips[i], clips[i + 1]
        u = (np.arange(2 * H) + 0.5) / (2 * H)
        fo, fi = np.cos(0.5 * np.pi * u), np.sin(0.5 * np.pi * u)
        xa = read_range(src, a.src_end - H, a.src_end + H)
        xb = read_range(src, b.src_start - H, b.src_start + H)
        ga = a.gains(np.arange(a.length - H, a.length + H, dtype=float))
        gb = b.gains(np.arange(-H, H, dtype=float))
        return (xa * (ga * fo)[:, None] + xb * (gb * fi)[:, None]).astype(np.float32)

    written = 0
    with open_writer(out_path, src) as w:
        for i, c in enumerate(clips):
            lx = i > 0 and disc(i - 1) and clips[i - 1].length >= 2 * H and c.length >= 2 * H
            rx = i < n - 1 and disc(i) and c.length >= 2 * H and clips[i + 1].length >= 2 * H
            if lx:
                x = xfade(i - 1)
                w.write(x)
                written += len(x)
            b0 = H if lx else 0
            b1 = c.length - (H if rx else 0)
            pos = b0
            for blk in iter_blocks(src, c.src_start + b0, c.src_start + b1):
                rel = np.arange(pos, pos + len(blk), dtype=float)
                g = c.gains(rel)
                # cut edges at the very start/end of the output get a short fade so a
                # mid-waveform start or end is not a click
                if i == 0 and c.src_start > 0:
                    g = g * np.clip(rel / (2 * H), 0, 1)
                if i == n - 1 and c.src_end < src.frames:
                    g = g * np.clip((c.length - rel) / (2 * H), 0, 1)
                out = blk if np.all(g == 1.0) else (blk * g[:, None].astype(np.float32))
                w.write(out)
                written += len(blk)
                pos += len(blk)
        clipped = w.clips
    if written != tl.length:
        raise AssertionError(f"render wrote {written} samples, timeline is {tl.length}")
    return {"path": str(out_path), "frames": written, "splices": tl.splices(), "clipped_samples": clipped}
