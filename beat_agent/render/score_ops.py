"""Typed chat operations on a recreation's score (the LLM never touches JS).

"mute the drums"        -> set_mute(track="drums", mute=true)
"make the bass brighter" -> brightness(track="bass", direction="up")
"play it at 140"        -> set_tempo(bpm=140)   (the number comes from the user)
Code applies them to the Score and the deterministic renderer rebuilds the page.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from beat_agent.transcribe.score import Score

BRIGHTNESS_STEP = {"small": 1.4, "medium": 2.0, "large": 3.0}   # cutoff multiplier per step (~half/one/1.6 octaves)


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SetMute(_M):
    op: Literal["set_mute"]
    track: str = Field(description="track id, name or role: drums, bass, keys, lead")
    mute: bool


class SetGain(_M):
    op: Literal["set_gain"]
    track: str
    db: float = Field(ge=-30, le=12, description="gain change in dB from the user's words")


class Brightness(_M):
    op: Literal["brightness"]
    track: str
    direction: Literal["up", "down"]
    amount: Literal["small", "medium", "large"] = "medium"


class SetTempo(_M):
    op: Literal["set_tempo"]
    bpm: Optional[float] = Field(None, ge=30, le=300, description="the tempo the user asked for; null = back to the analysed tempo")


class Clarify(_M):
    op: Literal["clarify"]
    question: str


class Reject(_M):
    op: Literal["reject"]
    reason: str


ScoreOp = Annotated[Union[SetMute, SetGain, Brightness, SetTempo, Clarify, Reject], Field(discriminator="op")]


class ScorePlan(_M):
    reading: str
    ops: list[ScoreOp] = Field(min_length=1)


class ScoreOpError(ValueError):
    pass


def _track(score: Score, ref: str):
    t = score.track(ref)
    if t is None:
        names = ", ".join(f"{x.id} ({x.role})" for x in score.tracks)
        raise ScoreOpError(f"There is no '{ref}' track in this recreation (tracks: {names}).")
    return t


def apply(score: Score, plan: ScorePlan) -> tuple[Score, str, str]:
    """Returns (new score, status, echo). Atomic: on any error the score is unchanged."""
    for o in plan.ops:
        if isinstance(o, Clarify):
            return score, "clarify", o.question
        if isinstance(o, Reject):
            return score, "rejected", o.reason
    s = score.model_copy(deep=True)
    lines = []
    try:
        for o in plan.ops:
            if isinstance(o, SetMute):
                t = _track(s, o.track)
                t.mute = o.mute
                lines.append(f"{t.name} {'muted' if o.mute else 'unmuted'}.")
            elif isinstance(o, SetGain):
                t = _track(s, o.track)
                new = max(-60.0, min(12.0, t.gain_db + o.db))
                if new != t.gain_db + o.db:
                    raise ScoreOpError(f"{t.name} gain would be {t.gain_db + o.db:+.1f} dB; the range is -60 to +12 dB.")
                lines.append(f"{t.name} gain {t.gain_db:+.1f} -> {new:+.1f} dB.")
                t.gain_db = new
            elif isinstance(o, Brightness):
                t = _track(s, o.track)
                if t.role == "drums":
                    raise ScoreOpError("Drums have no filter to brighten in this synth; try the gain or mute instead.")
                f = BRIGHTNESS_STEP[o.amount] ** (1 if o.direction == "up" else -1)
                c0 = t.patch.cutoff_hz
                c1 = max(40.0, min(18000.0, c0 * f))
                msg = f"{t.name} low-pass cutoff {c0:.0f} -> {c1:.0f} Hz"
                t.patch.cutoff_hz = c1
                if t.patch.osc == "fm":
                    i0 = t.patch.fm_index
                    t.patch.fm_index = max(0.0, min(20.0, i0 * f))
                    msg += f", FM index {i0:.2f} -> {t.patch.fm_index:.2f}"
                lines.append(msg + f" ({o.amount} step {o.direction}).")
            elif isinstance(o, SetTempo):
                old = s.tempo_bpm
                s.tempo_bpm = s.analysed_tempo_bpm if o.bpm is None else float(o.bpm)
                lines.append(f"Tempo {old:.2f} -> {s.tempo_bpm:.2f} BPM (loop {s.loop_beats * 60 / old:.3f} -> "
                             f"{s.loop_seconds():.3f} s); notes keep their beat positions, so rhythm is unchanged.")
    except ScoreOpError as e:
        return score, "error", f"{e} Nothing was changed."
    s.edit_history = list(score.edit_history) + [{"reading": plan.reading, "ops": [o.model_dump() for o in plan.ops]}]
    return s, "applied", "\n".join([f"Reading: {plan.reading}"] + lines)
