"""Text -> validated EditPlan -> EditSession. The LLM interprets; code executes.

The model sees the grid *summary* (counts, meter, tempo segments -- no beat
times) and recent turns, and must answer with JSON matching EditPlan. Output
is validated with pydantic; on failure we retry once with the validation
error; if that fails too, the user gets an error and nothing is executed.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from beat_agent.agent.llm import LLMError, LLMProvider, UsageLedger
from beat_agent.edit import ops as O
from beat_agent.edit.session import EditSession, TurnResult

SYSTEM = """You translate a music producer's chat message into edit operations on a beat grid.
You only choose operations and musical addresses. You never state or compute times in seconds,
tempo values, or anything measured from the audio -- the code measures and executes.

Addressing conventions (they are also stated to the user):
- Beats are 0-based positions. Beat 0 is the first downbeat. A beat range {"unit":"beats","start":a,"end":b}
  is half-open: from beat a up to but NOT including beat b. "beat 16-32" / "beats 16 to 32" -> start 16, end 32.
- Bars are 1-based. A bar range {"unit":"bars","start":a,"end":b} is inclusive. "bar 5" -> start 5, end 5.
- "the last N beats/bars" -> last_beats / last_bars (the code works out where they are).
- "the bar(s) before the tempo change" -> bars_before_tempo_change (count = number of bars). Use it even if the
  grid summary shows no tempo change: the code will tell the user. Set change only if the user named one.
- "the first N bars" -> bars 1..N.
- Points (for move / insert): before_beat, before_bar, after_bar, start, end.

Readings (state the one you used in `reading`, in plain words):
- "repeat X twice" / "play X twice" -> repeat with times_total = 2 (two plays in total, in place). "cut X and repeat
  it" selects X; it does not delete it. "three times" -> times_total 3.
- "loop X N times" -> times_total = N (N plays in total), then the track continues as before.
- "fade out the last N beats" -> fade, direction out, range last_beats N.
- Gains: use the dB the user says; if they give none ("a bit louder"), use clarify to ask for an amount.
- "undo" / "redo" -> undo / redo alone in the plan.
- Grid corrections (no audio change): "shift the downbeat by one beat" -> shift_downbeat beats=1 (later) or -1 if they
  say earlier; "it's in 3/4" -> set_beats_per_bar 3 (6/8 -> 6 if beats are eighths, else 2; use the grid summary's
  beat_unit); "the tempo is double that" -> set_tempo_octave 2.0; "half tempo" / "it's half that" -> 0.5.
- Vague or subjective targets ("the boring part", "the bad bit", "the best section") cannot be measured: use
  propose_cut (it measures a candidate and asks the user) -- never remove. Set bars only if the user said a length.
- After a proposal, a confirmation ("yes", "do it", "cut it") -> confirm_proposal. A refusal -> clarify.
- If numbers are out of range, still emit exactly what the user asked; the code checks ranges and tells the user.
  Never shrink, clamp or "fix" a range yourself.
- If the request is ambiguous in a way that changes the result (e.g. it is unclear whether numbers are beats or
  bars), use clarify with one short question. If it is not an audio-editing request, use reject.
- Several ranges in one message refer to the CURRENT numbering: emit the ops from the latest position in the track
  to the earliest, so earlier edits do not shift the numbers of later ones.

Examples (format only):
"drop beats 4 to 12" -> {"reading":"beats 4 up to (not incl.) 12","ops":[{"op":"remove","range":{"unit":"beats","start":4,"end":12}}]}
"make bars 9-12 play three times" -> {"reading":"three plays in total","ops":[{"op":"repeat","range":{"unit":"bars","start":9,"end":12},"times_total":3}]}
"move bar 2 to the end" -> {"reading":"bar 2 moved after the last bar","ops":[{"op":"move","range":{"unit":"bars","start":2,"end":2},"to":{"point":"end"}}]}
"turn the first 2 bars down 6 dB" -> {"reading":"bars 1-2, -6 dB","ops":[{"op":"gain","range":{"unit":"bars","start":1,"end":2},"db":-6}]}
"remove the weird part" -> {"reading":"'weird' is not measurable; propose a measured candidate","ops":[{"op":"propose_cut"}]}
"""


def gemini_schema() -> dict:
    return gemini_schema_for(O.EditPlan)


def gemini_schema_for(model) -> dict:
    """A pydantic model's JSON schema in the subset Gemini's response_json_schema
    accepts: refs inlined, oneOf -> anyOf, const -> enum, no discriminator/title."""
    s = model.model_json_schema()
    defs = s.pop("$defs", {})

    def fix(n):
        if isinstance(n, dict):
            if "$ref" in n:
                return fix(copy.deepcopy(defs[n["$ref"].split("/")[-1]]))
            n = {k: v for k, v in n.items() if k not in ("discriminator", "title")}
            if "oneOf" in n:
                n["anyOf"] = n.pop("oneOf")
            if "const" in n:
                n["enum"] = [n.pop("const")]
            return {k: fix(v) for k, v in n.items()}
        if isinstance(n, list):
            return [fix(x) for x in n]
        return n
    return fix(s)


@dataclass
class AgentTrace:
    user: str
    attempts: list = field(default_factory=list)   # [{raw, error, usage, cached}]
    plan: dict | None = None
    result: dict | None = None


class Agent:
    def __init__(self, llm: LLMProvider, session: EditSession):
        self.llm = llm
        self.session = session
        self.ledger = UsageLedger()
        self.schema = gemini_schema()
        self.traces: list[AgentTrace] = []

    def _context(self, text: str, error: str | None, prev_raw: str | None) -> str:
        g = self.session.grid()
        ctx = {
            "grid_summary": g.summary(),
            "beat_unit": g.time_signature.beat_unit,
            "pending_proposal": self.session.pending and {"bars": self.session.pending["bars"],
                                                          "criterion": self.session.pending["criterion"]},
            "can_undo": len(self.session.undo_stack), "can_redo": len(self.session.redo_stack),
            "recent_turns": [{"user": t["user"], "status": t["status"], "ops": t["ops"]} for t in self.session.turns[-6:]],
        }
        msg = f"STATE:\n{json.dumps(ctx)}\n\nUSER MESSAGE:\n{text}"
        if error:
            msg += (f"\n\nYour previous answer was rejected by the validator:\n{prev_raw}\nError:\n{error}\n"
                    "Answer again with valid JSON for the schema.")
        return msg

    def handle(self, text: str) -> tuple[TurnResult, AgentTrace]:
        tr = AgentTrace(user=text)
        self.traces.append(tr)
        err, raw = None, None
        plan = None
        for attempt in range(2):
            try:
                r = self.llm.complete_json(SYSTEM, self._context(text, err, raw), self.schema)
            except LLMError as e:
                res = TurnResult("error", f"The language model is unavailable: {e}. Nothing was changed.")
                tr.result = res.to_dict()
                return res, tr
            self.ledger.add(r)
            raw = r.text
            try:
                plan = O.EditPlan.model_validate_json(raw)
                tr.attempts.append({"raw": raw, "error": None, "usage": r.usage, "cached": r.cached})
                break
            except ValidationError as e:
                err = str(e)[:1500]
                tr.attempts.append({"raw": raw, "error": err, "usage": r.usage, "cached": r.cached})
                plan = None
        if plan is None:
            res = TurnResult("error", "I could not turn that into a valid edit (the model's answer failed validation "
                             "twice). Nothing was changed. Try rephrasing with beat or bar numbers.")
            self.session.turns.append({"user": text, "reading": "", "status": "error", "ops": [], "message": res.message})
            tr.result = res.to_dict()
            return res, tr
        tr.plan = plan.model_dump()
        res = self.session.apply(plan, text)
        tr.result = res.to_dict()
        return res, tr
