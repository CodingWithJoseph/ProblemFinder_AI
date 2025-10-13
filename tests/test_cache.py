from pathlib import Path

from problemfinder.core.cache import CacheConfig, ResponseCache, cached_api_call


def test_cached_api_call_round_trip(tmp_path: Path):
    config = CacheConfig(enabled=True, path=tmp_path / "cache.json", ttl_hours=1)
    cache = ResponseCache(config)

    calls = {"count": 0}

    def _fetch():
        calls["count"] += 1
        return {"payload": {"value": "foo"}}

    first = cached_api_call(
        model="gpt-4o",
        prompt_version="test",
        text="hello",
        cache=cache,
        fetch_fn=_fetch,
    )
    second = cached_api_call(
        model="gpt-4o",
        prompt_version="test",
        text="hello",
        cache=cache,
        fetch_fn=_fetch,
    )

    assert first == second
    assert calls["count"] == 1
