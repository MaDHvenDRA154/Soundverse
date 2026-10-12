"""Typed edit operations (the EDL) and the plan the LLM must return.

Ranges are musical addresses only. The LLM never supplies seconds or samples;
code resolves every address against the *current* grid (see resolve.py).
Conventions (also in the grid JSON and DECISIONS.md):
  beats are 0-based positions, beat ranges half-open:  beat 16-32 = beats 16..31
  bars are 1-based units, bar ranges inclusive:         bars 5-8 = 4 bars
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------- addresses
class BeatRange(_M):
    unit: Literal["beats"]
    start: int = Field(description="first beat index (0-based), included")
    end: int = Field(description="beat index where the range stops, NOT included (half-open)")


class BarRange(_M):
    unit: Literal["bars"]
    start: int = Field(description="first bar number (1-based), included")
    end: int = Field(description="last bar number, included. A single bar: start == end")


class LastBeats(_M):
    unit: Literal["last_beats"]
    count: int = Field(ge=1, description="the last N beats of the current track, through to the end of the audio")


class LastBars(_M):
    unit: Literal["last_bars"]
    count: int = Field(ge=1)


class BarsBeforeTempoChange(_M):
    unit: Literal["bars_before_tempo_change"]
    count: int = Field(1, ge=1, description="how many bars immediately before the change")
    change: Optional[int] = Field(None, ge=1, description="1-based index of the tempo change if the track has several; null if the user did not say")


class WholeTrack(_M):
    unit: Literal["whole_track"]


Range = Annotated[Union[BeatRange, BarRange, LastBeats, LastBars, BarsBeforeTempoChange, WholeTrack],
                  Field(discriminator="unit")]


class AtBeat(_M):
    point: Literal["before_beat"]
    beat: int


class AtBar(_M):
    point: Literal["before_bar"]
    bar: int


class AfterBar(_M):
    point: Literal["after_bar"]
    bar: int


class AtStart(_M):
    point: Literal["start"]


class AtEnd(_M):
    point: Literal["end"]


Point = Annotated[Union[AtBeat, AtBar, AfterBar, AtStart, AtEnd], Field(discriminator="point")]


# ----------------------------------------------------------------------- ops
class Remove(_M):
    op: Literal["remove"]
    range: Range


class KeepOnly(_M):
    op: Literal["keep_only"]
    range: Range


class Repeat(_M):
    op: Literal["repeat"]
    range: Range
    times_total: int = Field(ge=2, description="how many times the range plays IN TOTAL, in place (2 = original + 1 copy)")


class Move(_M):
    op: Literal["move"]
    range: Range
    to: Point


class Insert(_M):
    op: Literal["insert_copy"]
    range: Range
    at: Point
    copies: int = Field(1, ge=1)


class Fade(_M):
    op: Literal["fade"]
    range: Range
    direction: Literal["in", "out"]
    shape: Literal["equal_power", "linear"] = "equal_power"


class Gain(_M):
    op: Literal["gain"]
    range: Range
    db: float = Field(ge=-60, le=24, description="gain change in dB, taken from the user's words")


class ShiftDownbeat(_M):
    """Grid correction: move every bar line by N beats (positive = later)."""
    op: Literal["shift_downbeat"]
    beats: int


class SetBeatsPerBar(_M):
    """Grid correction: the user says the meter is different (e.g. 'it's in 3/4')."""
    op: Literal["set_beats_per_bar"]
    beats_per_bar: int = Field(ge=2, le=12)


class SetTempoOctave(_M):
    """Grid correction: 'the tempo is double that' -> 2.0, 'use half tempo' -> 0.5."""
    op: Literal["set_tempo_octave"]
    factor: Literal[2.0, 0.5]


class ProposeCut(_M):
    """For vague requests ('cut the boring part'): measure a candidate, do not cut."""
    op: Literal["propose_cut"]
    criterion: Literal["lowest_energy"] = "lowest_energy"
    bars: Optional[int] = Field(None, ge=1, description="length of the stretch in bars, only if the user said")


class ConfirmProposal(_M):
    op: Literal["confirm_proposal"]


class Undo(_M):
    op: Literal["undo"]
    turns: int = Field(1, ge=1)


class Redo(_M):
    op: Literal["redo"]
    turns: int = Field(1, ge=1)


class Clarify(_M):
    op: Literal["clarify"]
    question: str


class Reject(_M):
    op: Literal["reject"]
    reason: str


Op = Annotated[Union[Remove, KeepOnly, Repeat, Move, Insert, Fade, Gain, ShiftDownbeat, SetBeatsPerBar,
                     SetTempoOctave, ProposeCut, ConfirmProposal, Undo, Redo, Clarify, Reject],
               Field(discriminator="op")]

AUDIO_OPS = (Remove, KeepOnly, Repeat, Move, Insert, Fade, Gain)
GRID_OPS = (ShiftDownbeat, SetBeatsPerBar, SetTempoOctave)


class EditPlan(_M):
    reading: str = Field(description="one sentence: how you read the request, naming any convention or "
                                     "ambiguity (e.g. 'twice = two plays in total')")
    ops: list[Op] = Field(min_length=1)
