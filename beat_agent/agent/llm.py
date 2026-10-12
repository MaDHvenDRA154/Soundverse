"""LLM provider interface + the Gemini implementation.

Only `complete_json` is needed: system prompt + user content + a JSON schema
in, raw JSON text + token usage out. Swap providers by implementing
`LLMProvider`. Responses are cached on disk keyed by (model, system, contents,
schema), so eval re-runs cost nothing; the cache key covers the grid state
because the grid summary is part of the contents.

The cache lives in evals/llm_cache/ and is committed: a fresh clone replays the
exact answers behind the committed results.
  LLM_CACHE=on (default)  replay cached answers; a prompt not in the cache is sent
                          to the API (needs GEMINI_API_KEY) and its answer cached
  LLM_CACHE=replay        replay only: no API key needed, no call is ever made, a
                          prompt not in the cache is an error (the clean-clone check)
  LLM_CACHE=off           never read the cache: every call goes to the API with your
                          key; fresh answers replace the cached ones
BEAT_AGENT_LLM_CACHE_DIR overrides the directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


class LLMError(RuntimeError):
    """User-facing LLM failure (quota, network, bad model)."""


@dataclass
class LLMResponse:
    text: str
    usage: dict = field(default_factory=dict)   # prompt_tokens, output_tokens, thinking_tokens
    cached: bool = False
    model: str = ""
    latency_s: float = 0.0
    fetched_at: str = ""   # UTC time the answer came from the API (for a cached answer: when it was cached)
    cache_key: str = ""


def cache_mode() -> str:
    """'on' (default), 'replay' (cache only, no API) or 'off' (bypass reads), from LLM_CACHE."""
    v = os.environ.get("LLM_CACHE", "on").strip().lower()
    if v not in ("on", "replay", "off"):
        raise LLMError(f"LLM_CACHE must be 'on', 'replay' or 'off', not {v!r}")
    return v


class LLMProvider:
    model: str = "?"

    def complete_json(self, system: str, contents: str, schema: dict) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


# ------------------------------------------------------------------ costs
def price_per_mtok() -> tuple[float, float]:
    """USD per 1M (input, output+thinking) tokens, as an upper bound for any Gemini 3
    Flash / Flash-Lite model: the highest listed Flash price (Gemini 3.6/3.8 Flash
    from 2027-01-01: $1.50 / $7.50; checked 2026-10-12). Dev runs used
    gemini-3.1-flash-lite on the free tier, where the real charge is $0, so the
    reported figure is what the same calls would cost at most on a paid key.
    Override with GEMINI_PRICE_IN / GEMINI_PRICE_OUT."""
    return float(os.environ.get("GEMINI_PRICE_IN", "1.50")), float(os.environ.get("GEMINI_PRICE_OUT", "7.50"))


@dataclass
class UsageLedger:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        if r.cached:
            self.cached_calls += 1
            return  # a cached answer costs nothing on this run
        self.prompt_tokens += r.usage.get("prompt_tokens", 0)
        self.output_tokens += r.usage.get("output_tokens", 0)
        self.thinking_tokens += r.usage.get("thinking_tokens", 0)

    def cost_usd(self) -> float:
        pin, pout = price_per_mtok()
        return (self.prompt_tokens * pin + (self.output_tokens + self.thinking_tokens) * pout) / 1e6

    def to_dict(self) -> dict:
        pin, pout = price_per_mtok()
        return {"calls": self.calls, "cached_calls": self.cached_calls, "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens, "thinking_tokens": self.thinking_tokens,
                "est_cost_usd": round(self.cost_usd(), 5),
                "price_assumption_usd_per_mtok": {"input": pin, "output_incl_thinking": pout,
                                                  "note": "upper-bound estimate; see llm.price_per_mtok"}}


# ------------------------------------------------------------------ cache
def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _cache_dir() -> Path:
    """evals/llm_cache/ of this checkout (committed), or BEAT_AGENT_LLM_CACHE_DIR."""
    env = os.environ.get("BEAT_AGENT_LLM_CACHE_DIR")
    if env:
        d = Path(env)
    else:
        repo = Path(__file__).resolve().parents[2]
        d = (repo if (repo / "evals").is_dir() else Path.cwd()) / "evals" / "llm_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(model: str, system: str, contents: str, schema: dict) -> str:
    blob = json.dumps({"m": model, "s": system, "c": contents, "j": schema}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# ------------------------------------------------------------------ Gemini
FORBIDDEN_MODEL_PREFIXES = ("gemini-2.5", "gemini-2.0", "gemini-1.")


# The model every committed result was produced with (free tier, its own daily quota).
# Any Gemini 3 model works: set GEMINI_MODEL.
DEFAULT_MODEL = "gemini-3.1-flash-lite"


class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None, api_key: str | None = None, check_model: bool = True,
                 use_cache: bool = True):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        if self.model.startswith(FORBIDDEN_MODEL_PREFIXES):
            raise LLMError(f"Model {self.model!r} is not allowed (Gemini 2.x/1.x are retired or retiring); "
                           "set GEMINI_MODEL to a Gemini 3 model.")
        self.use_cache = use_cache
        if cache_mode() == "replay":   # cache only: no key, no client, no models.list
            self.client = None
            return
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set (put it in .env; see .env.example), or set LLM_CACHE=replay "
                           "to replay the committed answers without a key.")
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.use_cache = use_cache
        if check_model:
            self._check_model()

    def _check_model(self) -> None:
        try:
            names = {m.name.removeprefix("models/") for m in self.client.models.list()}
        except Exception as e:
            raise LLMError(f"Could not list Gemini models (key or network problem): {e}") from e
        if self.model not in names:
            # no fallback to an alias such as gemini-flash-latest: it may point at a non-Gemini-3 model
            g3 = sorted(n for n in names if n.startswith("gemini-3"))
            raise LLMError(f"Model {self.model!r} is not available for this key. Gemini 3 models it can use: {g3}")

    def complete_json(self, system: str, contents: str, schema: dict) -> LLMResponse:
        key = cache_key(self.model, system, contents, schema)
        cpath = _cache_dir() / f"{key}.json"
        mode = cache_mode()
        if self.use_cache and mode in ("on", "replay") and cpath.exists():
            d = json.loads(cpath.read_text())
            return LLMResponse(d["text"], d.get("usage", {}), cached=True, model=self.model,
                               fetched_at=d.get("fetched_at", ""), cache_key=key)
        if mode == "replay" or self.client is None:
            raise LLMError(f"no cached answer for this prompt in {_cache_dir()} (LLM_CACHE=replay makes no API calls; "
                           "the analysis or prompt changed since the cache was made)")
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )
        last = None
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
            except Exception as e:  # quota / transient
                last = e
                msg = str(e)
                if "PerDay" in msg:   # a daily quota will not recover in seconds; fail fast and say so
                    import re
                    delay = re.findall(r"'retryDelay': '(\d+)s'", msg)
                    when = f" (resets in about {int(delay[0]) // 3600} h)" if delay else ""
                    raise LLMError(f"daily Gemini quota for {self.model} is used up{when}. Cached sessions still replay; "
                                   "use another key or a model with its own quota via GEMINI_MODEL") from e
                if any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500")) and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise LLMError(f"Gemini call failed: {msg[:300]}") from e
            um = resp.usage_metadata
            usage = {"prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                     "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
                     "thinking_tokens": getattr(um, "thoughts_token_count", 0) or 0}
            text = resp.text or ""
            now = _utc(time.time())
            out = LLMResponse(text, usage, cached=False, model=self.model, latency_s=time.perf_counter() - t0,
                              fetched_at=now, cache_key=key)
            if self.use_cache and text:
                cpath.write_text(json.dumps({"text": text, "usage": usage, "model": self.model, "fetched_at": now}))
            return out
        raise LLMError(f"Gemini call failed after retries: {last}")
