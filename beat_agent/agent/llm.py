"""LLM provider interface + the Gemini implementation.

Only `complete_json` is needed: system prompt + user content + a JSON schema
in, raw JSON text + token usage out. Swap providers by implementing
`LLMProvider`. Responses are cached on disk keyed by (model, system, contents,
schema), so eval re-runs cost nothing; the cache key covers the grid state
because the grid summary is part of the contents.

Cache modes (GeminiProvider(cache=...)):
  "none"    live calls, no cache at all. The default for interactive use (the
            preview app and `beat-agent chat`), which always uses your key.
  "replay"  answers only from the cache directory; no key, no API call, a
            prompt not in the cache is an error. The eval harness default (it
            passes its committed cache directory).
  "on"      replay, and send a prompt not in the cache to the API, caching it.
  "off"     every call goes to the API; fresh answers replace the cached ones.
The eval harness reads its mode from LLM_CACHE (default "replay"; `--live` = "off").
LLM_CACHE does not affect the app or the CLI. The cache directory is passed in
by the caller (cache_dir=...); this package has no default location for it.
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


CACHE_MODES = ("none", "replay", "on", "off")
NO_KEY_INTERACTIVE = ("Chat needs a Gemini API key: put GEMINI_API_KEY=... in .env (copy .env.example), then "
                      "restart. Free keys: https://aistudio.google.com/apikey. The committed eval results replay "
                      "without a key (`make eval`).")


def cache_mode(default: str = "replay") -> str:
    """The eval harness's cache mode: LLM_CACHE, else `default` ('replay')."""
    v = os.environ.get("LLM_CACHE", default).strip().lower()
    if v not in ("replay", "on", "off"):
        raise LLMError(f"LLM_CACHE must be 'replay', 'on' or 'off', not {v!r}")
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


def _cache_dir(d) -> Path:
    d = Path(d)
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
                 cache: str = "none", cache_dir: str | os.PathLike | None = None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        if self.model.startswith(FORBIDDEN_MODEL_PREFIXES):
            raise LLMError(f"Model {self.model!r} is not allowed (Gemini 2.x/1.x are retired or retiring); "
                           "set GEMINI_MODEL to a Gemini 3 model.")
        if cache not in CACHE_MODES:
            raise LLMError(f"cache mode must be one of {CACHE_MODES}, not {cache!r}")
        if cache != "none" and cache_dir is None:
            raise LLMError(f"cache mode {cache!r} needs a cache_dir")
        self.cache, self.cache_dir = cache, cache_dir
        if cache == "replay":   # committed answers only: no key, no client, no models.list
            self.client = None
            return
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError(NO_KEY_INTERACTIVE if cache == "none" else
                           f"GEMINI_API_KEY is not set (put it in .env; see .env.example). LLM_CACHE={cache} makes "
                           "API calls; the default eval mode (replay) needs no key.")
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=key)
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
        cpath = _cache_dir(self.cache_dir) / f"{key}.json" if self.cache != "none" else None
        mode = self.cache
        if mode in ("on", "replay") and cpath.exists():
            d = json.loads(cpath.read_text())
            return LLMResponse(d["text"], d.get("usage", {}), cached=True, model=self.model,
                               fetched_at=d.get("fetched_at", ""), cache_key=key)
        if mode == "replay" or self.client is None:
            raise LLMError(f"no cached answer for this prompt in {self.cache_dir} (LLM_CACHE=replay makes no API calls; "
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
            if mode in ("on", "off") and text:
                cpath.write_text(json.dumps({"text": text, "usage": usage, "model": self.model, "fetched_at": now}))
            return out
        raise LLMError(f"Gemini call failed after retries: {last}")
