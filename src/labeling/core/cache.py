"""Response caching utilities used by the classification pipeline."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class CacheConfig:
    """Configuration for :class:`ResponseCache`."""

    enabled: bool = False
    path: Path = Path("data/.cache/responses.json")
    ttl_hours: int = 24

    @property
    def ttl_seconds(self) -> int:
        """Return the configured cache TTL in seconds."""

        return int(self.ttl_hours * 3600)


class ResponseCache:
    """Simple persistent cache with TTL semantics."""

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._data: Dict[str, Dict[str, Any]] = {}
        if self._config.enabled:
            self._load()

    def _load(self) -> None:
        """Populate the in-memory cache from disk."""

        path = self._config.path
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                self._data = payload
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load cache from %studio", path)
            self._data = {}

    def _prune(self) -> None:
        """Remove entries older than the configured TTL."""

        if not self._config.enabled:
            return
        ttl = self._config.ttl_seconds
        if ttl <= 0:
            return
        cutoff = time.time() - ttl
        expired = [key for key, meta in self._data.items() if meta.get("timestamp", 0) < cutoff]
        for key in expired:
            self._data.pop(key, None)
        if expired:
            logger.debug("Pruned %d expired cache entries", len(expired))

    def _persist(self) -> None:
        """Write the current cache state to disk atomically."""

        if not self._config.enabled:
            return
        path = self._config.path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def make_key(self, *, model: str, prompt_version: str, text: str) -> str:
        """Create a deterministic cache key for ``text``."""

        normalised = " ".join(text.split())
        payload = json.dumps({"model": model, "prompt": prompt_version, "text": normalised}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return a cached payload for ``key`` if present."""

        if not self._config.enabled:
            return None
        with self._lock:
            self._prune()
            entry = self._data.get(key)
            if entry is None:
                return None
            return entry.get("value")

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Persist ``value`` under ``key`` in the cache."""

        if not self._config.enabled:
            return
        with self._lock:
            self._data[key] = {"timestamp": time.time(), "value": value}
            self._persist()


def cached_api_call(
    *,
    model: str,
    prompt_version: str,
    text: str,
    cache: ResponseCache,
    fetch_fn: Callable[[], Dict[str, Any]],
    cache_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Wrap ``fetch_fn`` with cache semantics."""

    key = cache.make_key(model=model, prompt_version=prompt_version, text=text)
    cached = cache.get(key)
    if cached is not None:
        if cache_stats is not None:
            cache_stats["hits"] = cache_stats.get("hits", 0) + 1
        logger.debug("Cache hit for key %studio", key)
        return cached

    if cache_stats is not None:
        cache_stats["misses"] = cache_stats.get("misses", 0) + 1

    result = fetch_fn()
    cache.set(key, result)
    return result
