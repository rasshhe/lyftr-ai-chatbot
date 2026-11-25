import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm.caching.caching import Cache

from app.config import KeyEntry


class SimulatedProviderError(Exception):
    def __init__(self, status_code: int, message: str = "Simulated provider failure") -> None:
        super().__init__(message)
        self.status_code = status_code


def parse_failure_header(header_value: Optional[str]) -> Dict[str, int]:
    allowed_statuses = {429, 500, 502, 503}
    rules: Dict[str, int] = {}
    if not header_value:
        return rules

    for part in header_value.split(","):
        token = part.strip()
        if not token or "=" not in token:
            continue
        key_part, status_part = token.split("=", 1)
        key_id = key_part.strip()
        try:
            code = int(status_part.strip())
        except ValueError:
            continue
        if key_id and code in allowed_statuses:
            rules[key_id] = code
    return rules


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    total_chars = sum(len(m.get("role", "")) + len(m.get("content", "")) + 8 for m in messages)
    return max(1, (total_chars + 3) // 4)


def build_cache_key(model: str, messages: List[Dict[str, str]]) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"lyftr_llm_cache:{digest}"


class LLMGateway:
    def __init__(
        self,
        keys: List[KeyEntry],
        default_model: str,
        fallback_model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        enable_failure_simulation: bool = False,
    ) -> None:
        self.keys = keys
        self.default_model = default_model
        self.fallback_model = fallback_model
        merged: List[str] = [default_model]
        for m in (fallback_models or []) + ([fallback_model] if fallback_model else []):
            if m and m not in merged:
                merged.append(m)
        self.models = merged
        self.enable_failure_simulation = enable_failure_simulation

        litellm.cache = Cache(type="local")
        litellm.enable_cache()
        self._custom_cache: Dict[str, Dict[str, Any]] = {}

    @property
    def cache_enabled(self) -> bool:
        return litellm.cache is not None

    def _get_cached_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if cache_key in self._custom_cache:
            return self._custom_cache[cache_key]

        if litellm.cache is not None and hasattr(litellm.cache, "cache_dict"):
            raw = litellm.cache.cache_dict.get(cache_key)
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return None
            if isinstance(raw, dict):
                return raw
        return None

    def _set_cached_entry(self, cache_key: str, entry: Dict[str, Any]) -> None:
        self._custom_cache[cache_key] = entry
        if litellm.cache is not None and hasattr(litellm.cache, "cache_dict"):
            litellm.cache.cache_dict[cache_key] = json.dumps(entry)

    def _extract_usage_and_text(
        self,
        response: Any,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, int, int]:
        reply_text = ""
        try:
            reply_text = response.choices[0].message.content or ""
        except Exception:
            if isinstance(response, dict):
                reply_text = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

        reply_text = reply_text.strip()
        if not reply_text:
            reply_text = "No response content returned by model."

        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")

        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None

        if usage is not None:
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
            else:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)

        if input_tokens is None:
            try:
                input_tokens = litellm.token_counter(model=self.default_model, messages=messages)
            except Exception:
                input_tokens = _estimate_prompt_tokens(messages)

        if output_tokens is None:
            output_tokens = _estimate_tokens(reply_text)

        return reply_text, int(input_tokens), int(output_tokens)

    def generate_reply(
        self,
        messages: List[Dict[str, str]],
        simulate_failure_header: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        start_perf = time.perf_counter()
        cache_key = build_cache_key(self.default_model, messages)

        sim_rules = (
            parse_failure_header(simulate_failure_header)
            if self.enable_failure_simulation
            else {}
        )

        cached = self._get_cached_entry(cache_key)
        if cached is not None and not sim_rules:
            latency_ms = max(1, int(round((time.perf_counter() - start_perf) * 1000)))
            metadata = {
                "resolution_source": "LLM_CACHE",
                "model": cached["model"],
                "latency_ms": latency_ms,
                "input_tokens": int(cached["input_tokens"]),
                "output_tokens": int(cached["output_tokens"]),
                "key_id": None,
                "fallback_attempts": 0,
            }
            return cached["reply"], metadata

        if not self.keys:
            raise RuntimeError("No LLM API keys configured")

        models_to_try = getattr(self, "models", None) or [self.default_model]
        fallback_attempts = 0
        last_error: Optional[Exception] = None

        for model_name in models_to_try:
            for key_entry in self.keys:
                sim_status = sim_rules.get(key_entry.key_id)
                if sim_status is None and "*" in sim_rules:
                    sim_status = sim_rules["*"]

                if sim_status is not None:
                    fallback_attempts += 1
                    last_error = SimulatedProviderError(
                        sim_status,
                        f"Simulated {sim_status} for {key_entry.key_id}",
                    )
                    continue

                try:
                    response = litellm.completion(
                        model=model_name,
                        messages=messages,
                        api_key=key_entry.api_key,
                        caching=True,
                        timeout=15.0,
                    )
                    reply_text, input_tokens, output_tokens = self._extract_usage_and_text(
                        response, messages
                    )
                    latency_ms = max(1, int(round((time.perf_counter() - start_perf) * 1000)))

                    cache_entry = {
                        "reply": reply_text,
                        "model": model_name,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                    self._set_cached_entry(cache_key, cache_entry)

                    metadata = {
                        "resolution_source": "LIVE_INFERENCE",
                        "model": model_name,
                        "latency_ms": latency_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "key_id": key_entry.key_id,
                        "fallback_attempts": fallback_attempts,
                    }
                    return reply_text, metadata

                except Exception as exc:
                    fallback_attempts += 1
                    last_error = exc
                    print(f"[LLMGateway] {key_entry.key_id} failed for model={model_name}: {type(exc).__name__}: {exc}", flush=True)
                    continue

        raise RuntimeError(f"All providers failed after {fallback_attempts} attempts") from last_error
