"""Chat edits on a recreation: text -> validated ScorePlan -> code applies it.

Same provider, validation and retry rules as the beat-grid agent; the model
never sees or writes JavaScript, only typed score operations.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from beat_agent.agent.agent import gemini_schema_for
from beat_agent.agent.llm import LLMError, LLMProvider, UsageLedger
from beat_agent.render import score_ops as S
from beat_agent.transcribe.score import Score

SYSTEM = """You translate a producer's chat message into operations on a synthesised recreation of a loop.
You only choose operations. You never write code and never state anything measured from the audio.
Operations: set_mute(track, mute), set_gain(track, db), brightness(track, up|down, small|medium|large),
set_tempo(bpm) -- bpm only as the user says it; null means "back to the original tempo".
Tracks are named by id/role: drums, bass, keys, lead (see STATE). "brighter" = brightness up; "darker",
"duller", "warmer" = down. "a bit" = small, "much" = large, otherwise medium. Gains need a number of dB from
the user; if there is none ("louder"), use clarify. If the track does not exist, still emit the op with the
user's word -- the code will say which tracks exist. Anything that is not one of these edits -> reject.
"""


class ScoreAgent:
    def __init__(self, llm: LLMProvider, score: Score):
        self.llm, self.score = llm, score
        self.ledger = UsageLedger()
        self.schema = gemini_schema_for(S.ScorePlan)
        self.history: list[dict] = []
        self.llm_error: str | None = None   # last provider failure (quota/network)

    def handle(self, text: str) -> tuple[str, str, dict | None]:
        ctx = {"tracks": [{"id": t.id, "role": t.role, "mute": t.mute, "gain_db": t.gain_db,
                           "cutoff_hz": round(t.patch.cutoff_hz)} for t in self.score.tracks],
               "tempo_bpm": self.score.tempo_bpm, "recent": self.history[-4:]}
        err = raw = None
        for _ in range(2):
            msg = f"STATE:\n{json.dumps(ctx)}\n\nUSER MESSAGE:\n{text}"
            if err:
                msg += f"\n\nYour previous answer failed validation:\n{raw}\n{err}\nAnswer again."
            try:
                r = self.llm.complete_json(SYSTEM, msg, self.schema)
            except LLMError as e:
                self.llm_error = str(e)
                return "error", f"The language model is unavailable: {e}. Nothing was changed.", None
            self.ledger.add(r)
            raw = r.text
            try:
                plan = S.ScorePlan.model_validate_json(raw)
                break
            except ValidationError as e:
                err, plan = str(e)[:1200], None
        if plan is None:
            return "error", "I could not turn that into a valid change (validation failed twice). Nothing was changed.", None
        self.score, status, echo = S.apply(self.score, plan)
        self.history.append({"user": text, "status": status, "ops": [o.model_dump() for o in plan.ops]})
        return status, echo, plan.model_dump()
