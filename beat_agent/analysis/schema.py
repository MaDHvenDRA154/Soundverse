"""Beat grid JSON schema (versioned). Shared by analysis, editing and sync."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

GRID_SCHEMA_VERSION = "1.0"

CONVENTIONS = {
    "time_unit": "seconds from the start of this audio file",
    "beat_index": "0-based. Beat 0 is the first downbeat (start of bar 1). Pickup beats before it are negative.",
    "bar": "1-based. Bar 1 starts at beat 0. Pickup beats belong to bar 0.",
    "beat_in_bar": "1-based position within the bar (1 = downbeat).",
    "beat_range": "half-open [start, end): 'beat 16-32' = beats 16..31 = 16 beats.",
    "bar_range": "inclusive: 'bars 5-8' = 4 bars, 'bar 5' = one bar.",
    "beat_level": "whatever pulse the tracker locked to; see time_signature.beat_unit and tempo.alternatives.",
    "no_beat_regions": "stretches with no detectable pulse; no beats are emitted there and indices skip nothing (they count emitted beats).",
}


class Beat(BaseModel):
    time: float
    index: int
    bar: int
    beat_in_bar: int
    confidence: float = Field(ge=0, le=1, description="tracker activation at this beat")
    region: int = 0


class Bar(BaseModel):
    bar: int
    start: float
    end: float
    n_beats: int
    complete: bool


class TempoSegment(BaseModel):
    kind: Literal["constant", "ramp", "free"]
    start_beat: int          # beat index, inclusive
    end_beat: int            # beat index, exclusive
    start_time: float
    end_time: float
    bpm_start: float
    bpm_end: float
    fit_residual_rms_ms: Optional[float] = None


class TempoAlternative(BaseModel):
    factor: float
    bpm: float
    onset_support: float = Field(description="x2: half-beat / on-beat onset strength; x0.5: weaker / stronger "
                                             "alternating-beat onset strength. Near 1 = that octave is as plausible as the tracked one.")


class Tempo(BaseModel):
    bpm: Optional[float]
    type: Literal["constant", "variable", "none"]
    segments: list[TempoSegment]
    changes: list[dict] = Field(default_factory=list, description="tempo step changes: beat index, time, from/to bpm")
    alternatives: list[TempoAlternative] = Field(default_factory=list)
    octave_correction: float = 1.0


class TimeSignature(BaseModel):
    numerator: Optional[int]
    denominator: Optional[int]
    beats_per_bar: Optional[int]
    beat_unit: Optional[str] = None     # quarter | eighth | dotted_quarter
    subdivision: Optional[Literal["binary", "ternary"]] = None
    subdivision_ratio: Optional[float] = None
    confidence: float = 0.0
    candidates: dict = Field(default_factory=dict)


class Region(BaseModel):
    start: float
    end: float


class Confidence(BaseModel):
    overall: float
    beat: float
    downbeat: float
    coverage: float
    how: str


class BeatGrid(BaseModel):
    schema_version: str = GRID_SCHEMA_VERSION
    conventions: dict = Field(default_factory=lambda: dict(CONVENTIONS))
    source: dict
    duration: float
    tempo: Tempo
    time_signature: TimeSignature
    beats: list[Beat]
    downbeats: list[float]
    bars: list[Bar]
    no_beat_regions: list[Region]
    confidence: Confidence
    messages: list[str] = Field(default_factory=list)
    analysis: dict = Field(default_factory=dict)
    edit_history: list[dict] = Field(default_factory=list)

    # ---------------------------------------------------------------- helpers
    def beat_times(self):
        import numpy as np
        return np.array([b.time for b in self.beats])

    def n_bars(self) -> int:
        return sum(1 for b in self.bars if b.bar >= 1)

    def bar(self, n: int) -> Optional[Bar]:
        for b in self.bars:
            if b.bar == n:
                return b
        return None

    def beat(self, index: int) -> Optional[Beat]:
        if not self.beats:
            return None
        off = index - self.beats[0].index
        if 0 <= off < len(self.beats) and self.beats[off].index == index:
            return self.beats[off]
        for b in self.beats:
            if b.index == index:
                return b
        return None

    def summary(self) -> dict:
        """Compact view for the LLM prompt: addresses only, no per-beat times."""
        ts = self.time_signature
        return {
            "duration_s": round(self.duration, 3),
            "n_beats": len(self.beats),
            "first_beat_index": self.beats[0].index if self.beats else None,
            "last_beat_index": self.beats[-1].index if self.beats else None,
            "n_bars": self.n_bars(),
            "has_pickup_bar": any(b.bar == 0 for b in self.bars),
            "time_signature": f"{ts.numerator}/{ts.denominator}" if ts.numerator else None,
            "beats_per_bar": ts.beats_per_bar,
            "tempo_type": self.tempo.type,
            "tempo_bpm": self.tempo.bpm,
            "tempo_segments": [{"kind": s.kind, "start_beat": s.start_beat, "end_beat": s.end_beat,
                                "bpm_start": s.bpm_start, "bpm_end": s.bpm_end} for s in self.tempo.segments],
            "tempo_changes": self.tempo.changes,
            "no_beat_regions_s": [[round(r.start, 2), round(r.end, 2)] for r in self.no_beat_regions],
        }
