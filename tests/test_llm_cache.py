"""LLM answer cache: default replay, LLM_CACHE=off bypass, provenance fields. No network."""
from types import SimpleNamespace

import pytest

from beat_agent.agent import llm as L


class FakeModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        um = SimpleNamespace(prompt_token_count=10, candidates_token_count=2, thoughts_token_count=1)
        return SimpleNamespace(text='{"ok": %d}' % self.calls, usage_metadata=um)


def provider(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "_cache_dir", lambda: tmp_path)
    p = L.GeminiProvider.__new__(L.GeminiProvider)   # skip key + models.list
    p.model, p.use_cache = "gemini-3-test", True
    p.client = SimpleNamespace(models=FakeModels())
    return p


def test_default_replays_and_off_bypasses(monkeypatch, tmp_path):
    p = provider(monkeypatch, tmp_path)
    monkeypatch.delenv("LLM_CACHE", raising=False)
    a = p.complete_json("sys", "hello", {"type": "object"})
    b = p.complete_json("sys", "hello", {"type": "object"})
    assert not a.cached and b.cached and b.text == a.text and p.client.models.calls == 1
    assert a.fetched_at and b.fetched_at == a.fetched_at          # replay reports the original fetch time
    monkeypatch.setenv("LLM_CACHE", "off")
    c = p.complete_json("sys", "hello", {"type": "object"})
    assert not c.cached and p.client.models.calls == 2 and c.text != a.text
    monkeypatch.setenv("LLM_CACHE", "on")
    assert p.complete_json("sys", "hello", {"type": "object"}).text == c.text   # the live answer is what replays next


def test_bad_cache_mode_is_an_error(monkeypatch):
    monkeypatch.setenv("LLM_CACHE", "maybe")
    with pytest.raises(L.LLMError):
        L.cache_mode()


def test_cache_entry_holds_no_credentials(monkeypatch, tmp_path):
    p = provider(monkeypatch, tmp_path)
    monkeypatch.delenv("LLM_CACHE", raising=False)
    p.complete_json("sys", "hello", {"type": "object"})
    import json
    (f,) = tmp_path.glob("*.json")
    assert set(json.loads(f.read_text())) == {"text", "usage", "model", "fetched_at"}
