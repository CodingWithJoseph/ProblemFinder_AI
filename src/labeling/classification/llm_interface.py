"""Thin wrapper around the OpenAI chat completion API for classification."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, MutableMapping

from openai import OpenAI

from labeling.classification.ensemble import EnsembleMemberResult, PROMPT_VERSION
from labeling.core.cache import ResponseCache, cached_api_call
from labeling.core.config import ModelConfig
from labeling.utils.rate_limit import RateLimiter

logger = logging.getLogger(__name__)


class OpenAIEnsembleFactory:
    """Factory that produces ensemble member callables backed by OpenAI."""

    def __init__(
        self,
        *,
        client: OpenAI,
        cache: ResponseCache,
        cache_stats: MutableMapping[str, int],
        rate_limiter: RateLimiter,
        model_cfg: ModelConfig,
    ) -> None:
        self._client = client
        self._cache = cache
        self._cache_stats = cache_stats
        self._rate_limiter = rate_limiter
        self._model_cfg = model_cfg

    def create_member(self, member_name: str) -> Callable[[str], EnsembleMemberResult]:
        """Return a callable that classifies text using the specified prompt style."""

        if member_name == "direct":
            system_prompt = (
                "You are a classifier for Reddit posts. "
                "Decide whether the post describes a problem, if that problem is solvable with software alone, and whether external help is needed. "
                "Respond ONLY with minified JSON in this form: {\"is_problem\": \"0 or 1\", \"is_software_solvable\": \"0 or 1\", \"is_external\": \"0 or 1\", \"confidence\": number between 0 and 1, \"rationale\": short factual string}. "
                "Keep the rationale concise and free of Markdown."
            )
            default_confidence = 0.8
        else:
            system_prompt = (
                "You are a careful classifier for Reddit posts. Think step by step about whether the post describes a problem, if software alone can solve it, and whether outside coordination is required. "
                "After reasoning internally, respond ONLY with JSON matching {\"is_problem\": \"0 or 1\", \"is_software_solvable\": \"0 or 1\", \"is_external\": \"0 or 1\", \"confidence\": number between 0 and 1, \"rationale\": short factual string}."
            )
            default_confidence = 0.9

        def _coerce_label(value: Any) -> str:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped in {"0", "1"}:
                    return stripped
                if stripped.lower() in {"true", "false"}:
                    return "1" if stripped.lower() == "true" else "0"
                if stripped.isdigit():
                    return "1" if int(stripped) >= 1 else "0"
            if isinstance(value, bool):
                return "1" if value else "0"
            if isinstance(value, (int, float)):
                return "1" if float(value) >= 0.5 else "0"
            return ""

        def _call(text: str) -> EnsembleMemberResult:
            prompt_version = f"{PROMPT_VERSION}:{member_name}"

            def _fetch() -> Dict[str, Any]:
                with self._rate_limiter.slot():
                    response = self._client.chat.completions.create(
                        model=self._model_cfg.name,
                        temperature=self._model_cfg.temperature,
                        seed=self._model_cfg.seed,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Post:\n{text.strip()}"},
                        ],
                    )
                content = ""
                if response.choices:
                    content = response.choices[0].message.content or ""
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    parts = cleaned.split("\n", 1)
                    if parts and parts[0].lower().startswith("json"):
                        cleaned = parts[1] if len(parts) > 1 else ""
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Failed to parse JSON from LLM response: {cleaned}") from exc

                rationale = str(parsed.get("rationale", "")).strip()
                confidence_value = parsed.get("confidence", default_confidence)
                try:
                    confidence_float = float(confidence_value)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    confidence_float = default_confidence
                confidence_float = max(0.0, min(1.0, confidence_float))

                payload = {
                    "intent": str(parsed.get("intent", "")) if parsed.get("intent") is not None else "",
                    "is_problem": _coerce_label(parsed.get("is_problem")),
                    "is_software_solvable": _coerce_label(parsed.get("is_software_solvable")),
                    "is_external": _coerce_label(parsed.get("is_external")),
                    "problem_reason": rationale,
                    "software_reason": rationale,
                    "external_reason": rationale,
                    "detected_patterns": "",
                    "confidence": confidence_float,
                }
                return {"payload": payload, "confidence": confidence_float, "rationale": rationale}

            response = cached_api_call(
                model=self._model_cfg.name,
                prompt_version=prompt_version,
                text=text,
                cache=self._cache,
                fetch_fn=_fetch,
                cache_stats=self._cache_stats,
            )
            payload = response.get("payload", {})
            confidence = response.get("confidence", default_confidence)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                confidence = default_confidence
            rationale = response.get("rationale", "")
            return EnsembleMemberResult(member=member_name, payload=payload, confidence=confidence, rationale=rationale)

        return _call


__all__ = ["OpenAIEnsembleFactory"]
