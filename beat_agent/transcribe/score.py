"""Score JSON: the contract between analysis and playback (versioned).

Notes carry their position in beats (what the player schedules, so a tempo
change is a single number) and in seconds of the source clip (what evals
compare with the original audio).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCORE_SCHEMA_VERSION = "1.0"

Role = Literal["drums", "bass", "keys", "lead"]
DrumKind = Literal["kick", "snare", "hat"]


class Note(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_beats: float = Field(description="from the loop start, in beats")
    dur_beats: float
    start_s: float = Field(description="seconds in the source clip (for evaluation)")
    dur_s: float
    pitch: int = Field(ge=0, le=127, description="MIDI pitch; drums use GM numbers 36/38/42")
    velocity: int = Field(ge=1, le=127)
    drum: Optional[DrumKind] = None


class Patch(BaseModel):
    """Synthesis parameters the deterministic renderer understands."""
    model_config = ConfigDict(extra="forbid")
    osc: Literal["saw", "square", "triangle", "sine", "fm"] = "saw"
    cutoff_hz: float = Field(1800.0, ge=40, le=18000)
    resonance: float = Field(1.0, ge=0.1, le=20)
    attack_s: float = Field(0.005, ge=0.0005, le=2)
    decay_s: float = Field(0.15, ge=0.001, le=4)
    sustain: float = Field(0.6, ge=0, le=1)
    release_s: float = Field(0.12, ge=0.005, le=4)
    fm_ratio: float = Field(2.0, ge=0.25, le=16)
    fm_index: float = Field(1.5, ge=0, le=20)
    detune_cents: float = Field(0.0, ge=0, le=50)


class Track(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    role: Role
    mute: bool = False
    gain_db: float = Field(0.0, ge=-60, le=12)
    patch: Patch = Field(default_factory=Patch)
    notes: list[Note]


class Score(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCORE_SCHEMA_VERSION
    tempo_bpm: float = Field(gt=20, lt=400)
    analysed_tempo_bpm: float
    time_signature: tuple[int, int]
    beats_per_bar: int
    bars: int = Field(ge=1, le=16)
    loop_beats: float
    clip: dict = Field(description="source path, the loop's start/end in seconds, bar range")
    tracks: list[Track]
    provenance: dict = Field(default_factory=dict)
    edit_history: list[dict] = Field(default_factory=list)

    def loop_seconds(self) -> float:
        return self.loop_beats * 60.0 / self.tempo_bpm

    def track(self, ref: str) -> Optional[Track]:
        ref = ref.lower()
        for t in self.tracks:
            if t.id == ref or t.name.lower() == ref or t.role == ref:
                return t
        return None
