"""Threaded processing helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Sequence, Tuple

from problemfinder.core.config import ParallelConfig


def parallel_process_batch(
    tasks: Sequence[Tuple[int, Any]],
    worker_fn: Callable[[int, Any], Any],
    *,
    parallel_cfg: ParallelConfig,
) -> List[Any]:
    """Run ``worker_fn`` over ``tasks`` in parallel while preserving order."""

    results: List[Any] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=parallel_cfg.max_workers) as executor:
        future_items: List[Tuple[int, Any]] = []
        for position, (index, payload) in enumerate(tasks):
            future_items.append((position, executor.submit(worker_fn, index, payload)))
        for position, future in future_items:
            try:
                results[position] = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                results[position] = exc
    return results
