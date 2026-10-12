"""Agent plumbing with a scripted provider (validation, retry, graceful errors).

The scripted provider exists only to test these code paths. The brief's chat
sessions are run through the real Gemini model by evals/part2.py.
"""
import json

import pytest

from beat_agent.agent.agent import Agent, gemini_schema
from beat_agent.agent.llm import LLMError, LLMProvider, LLMResponse
from beat_agent.analysis.pipeline import analyze
from beat_agent.edit.session import EditSession
from beat_agent.io.audio import open_source
from evals import synth


class Scripted(LLMProvider):
    model = "scripted"

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    def complete_json(self, system, contents, schema):
        self.seen.append(contents)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return LLMResponse(a, {"prompt_tokens": 100, "output_tokens": 10, "thinking_tokens": 5})


@pytest.fixture(scope="module")
def sess_factory(tmp_path_factory):
    d = tmp_path_factory.mktemp("ag")
    a, _ = synth.make_track(segments=((64, 120, 120),), seed=9)
    p = synth.write(d / "a.wav", a, 44100)
    src = open_source(p)
    g = analyze(p, src)
    return lambda: EditSession(src, g)


def test_schema_has_no_unsupported_keywords():
    s = json.dumps(gemini_schema())
    assert "$ref" not in s and "oneOf" not in s and '"const"' not in s


def test_valid_plan_is_applied_and_tokens_counted(sess_factory):
    s = sess_factory()
    ag = Agent(Scripted(['{"reading":"bar 2","ops":[{"op":"remove","range":{"unit":"bars","start":2,"end":2}}]}']), s)
    res, tr = ag.handle("remove bar 2")
    assert res.status == "applied" and len(s.undo_stack) == 1
    assert ag.ledger.prompt_tokens == 100 and ag.ledger.cost_usd() > 0


def test_invalid_then_valid_retries_once_with_the_error(sess_factory):
    s = sess_factory()
    llm = Scripted(['{"reading":"x","ops":[{"op":"explode"}]}',
                    '{"reading":"x","ops":[{"op":"remove","range":{"unit":"bars","start":2,"end":2}}]}'])
    res, tr = Agent(llm, s).handle("remove bar 2")
    assert res.status == "applied" and len(tr.attempts) == 2
    assert "rejected by the validator" in llm.seen[1]


def test_invalid_twice_executes_nothing(sess_factory):
    s = sess_factory()
    res, tr = Agent(Scripted(["not json", '{"ops":[]}']), s).handle("do a thing")
    assert res.status == "error" and not s.undo_stack


def test_seconds_are_not_accepted(sess_factory):
    s = sess_factory()
    bad = '{"reading":"x","ops":[{"op":"remove","range":{"unit":"seconds","start":1.5,"end":3.0}}]}'
    res, _ = Agent(Scripted([bad, bad]), s).handle("remove 1.5 to 3 seconds")
    assert res.status == "error" and not s.undo_stack


def test_llm_outage_is_graceful(sess_factory):
    s = sess_factory()
    res, _ = Agent(Scripted([LLMError("quota")]), s).handle("remove bar 2")
    assert res.status == "error" and "unavailable" in res.message


def test_forbidden_models_refused(monkeypatch):
    from beat_agent.agent.llm import GeminiProvider
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    for m in ("gemini-2.5-flash", "gemini-2.0-flash"):
        with pytest.raises(LLMError, match="not allowed"):
            GeminiProvider(model=m, check_model=False)
