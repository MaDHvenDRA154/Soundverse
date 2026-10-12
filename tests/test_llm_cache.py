"""LLM answer cache modes, per-caller defaults, committed cache contents. No network."""
import json
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


def provider(monkeypatch, tmp_path, cache):
    p = L.GeminiProvider.__new__(L.GeminiProvider)   # skip key + models.list
    p.model, p.cache, p.cache_dir = "gemini-3-test", cache, tmp_path
    p.client = SimpleNamespace(models=FakeModels())
    return p


def ask(p):
    return p.complete_json("sys", "hello", {"type": "object"})


def test_on_replays_and_off_bypasses(monkeypatch, tmp_path):
    p = provider(monkeypatch, tmp_path, "on")
    a, b = ask(p), ask(p)
    assert not a.cached and b.cached and b.text == a.text and p.client.models.calls == 1
    assert a.fetched_at and b.fetched_at == a.fetched_at and b.cache_key == a.cache_key
    p.cache = "off"
    c = ask(p)
    assert not c.cached and p.client.models.calls == 2 and c.text != a.text
    p.cache = "on"
    assert ask(p).text == c.text            # the live answer is what replays next


def test_none_is_live_and_writes_nothing(monkeypatch, tmp_path):
    p = provider(monkeypatch, tmp_path, "none")
    ask(p), ask(p)
    assert p.client.models.calls == 2 and not list(tmp_path.glob("*"))


def test_eval_default_is_replay_and_env_overrides(monkeypatch):
    monkeypatch.delenv("LLM_CACHE", raising=False)
    assert L.cache_mode() == "replay"
    monkeypatch.setenv("LLM_CACHE", "off")
    assert L.cache_mode() == "off"
    monkeypatch.setenv("LLM_CACHE", "maybe")
    with pytest.raises(L.LLMError):
        L.cache_mode()


def _no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)


def test_replay_needs_no_key_and_never_calls(monkeypatch, tmp_path):
    _no_key(monkeypatch)
    p = L.GeminiProvider(model="gemini-3-test", cache="replay", cache_dir=tmp_path)
    assert p.client is None
    with pytest.raises(L.LLMError, match="no cached answer"):
        ask(p)


def test_interactive_default_is_live_and_explains_a_missing_key(monkeypatch):
    _no_key(monkeypatch)
    monkeypatch.setenv("LLM_CACHE", "replay")          # must not leak into the app / CLI
    with pytest.raises(L.LLMError) as e:
        L.GeminiProvider()
    assert "GEMINI_API_KEY" in str(e.value) and "aistudio.google.com/apikey" in str(e.value)


def test_cli_chat_without_key_prints_the_message(monkeypatch, capsys):
    _no_key(monkeypatch)
    from beat_agent import cli
    rc = cli.main(["chat", "does-not-matter.wav"])
    assert rc == 2 and "Chat needs a Gemini API key" in capsys.readouterr().err


def test_cache_entry_fields(monkeypatch, tmp_path):
    ask(provider(monkeypatch, tmp_path, "on"))
    (f,) = tmp_path.glob("*.json")
    assert set(json.loads(f.read_text())) == {"text", "usage", "model", "fetched_at"}


def test_committed_cache_holds_no_secrets():
    from evals.cache_tool import check
    assert check() == []


def test_app_without_key_reports_the_message(monkeypatch):
    _no_key(monkeypatch)
    monkeypatch.setenv("LLM_CACHE", "replay")          # ignored by the app
    from beat_agent.app import server
    monkeypatch.setattr(server, "_llm", {"provider": None, "error": None, "tried": False})
    assert server.llm() is None and "Chat needs a Gemini API key" in server._llm["error"]
