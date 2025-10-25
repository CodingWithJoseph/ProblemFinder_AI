"""End-to-end pipeline orchestration for the ProblemFinder classifier."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
from jsonschema import ValidationError, validate
from openai import OpenAI

from labeling.classification.ensemble import ensemble_classify, summarise_member_agreement
from labeling.classification.llm_interface import OpenAIEnsembleFactory
from labeling.classification.rules import CLASSIFICATION_SCHEMA, Version2RuleEngine
from labeling.core.cache import ResponseCache
from labeling.core.config import DedupeConfig, RunConfig, SplitConfig
from labeling.core.dedupe import deduplicate_dataframe, write_dedupe_report
from labeling.reporting.evaluation import compare_against_history, evaluate_against_gold
from labeling.reporting.summary import generate_summary_report
from labeling.utils.concurrency import parallel_process_batch
from labeling.utils.logging import structured_log
from labeling.utils.rate_limit import RateLimiter
from labeling.utils.redaction import redact_pii

logger = logging.getLogger(__name__)


def assign_splits(
    df: pd.DataFrame,
    cluster_members: Dict[str, List[str]],
    config: SplitConfig,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test splits ensuring clusters remain together."""

    if not config.enabled:
        df["split"] = "unsplit"
        return df

    ratios = [config.train_ratio, config.val_ratio, config.test_ratio]
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0")

    canonical_ids = list(cluster_members.keys())
    random.Random(seed).shuffle(canonical_ids)

    total = len(canonical_ids)
    train_cut = int(total * config.train_ratio)
    val_cut = train_cut + int(total * config.val_ratio)

    assignments: Dict[str, str] = {}
    for i, canonical_id in enumerate(canonical_ids):
        if i < train_cut:
            split = "train"
        elif i < val_cut:
            split = "val"
        else:
            split = "test"
        assignments[canonical_id] = split

    df["split"] = df["canonical_post_id"].map(assignments)
    return df


def _normalise_config(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _normalise_config(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalise_config(item) for item in payload]
    if isinstance(payload, Path):
        return str(payload)
    return payload


def _create_member_callables(
    *,
    engine: Version2RuleEngine,
    cache: ResponseCache,
    cache_stats: MutableMapping[str, int],
    rate_limiter: RateLimiter,
    model_cfg,
    client: OpenAI,
) -> Dict[str, Any]:
    factory = OpenAIEnsembleFactory(
        client=client,
        cache=cache,
        cache_stats=cache_stats,
        rate_limiter=rate_limiter,
        model_cfg=model_cfg,
    )

    members: Dict[str, Any] = {"rules": lambda text: None}
    members["direct"] = factory.create_member("direct")
    members["reasoning"] = factory.create_member("reasoning")
    return members


def classify_dataframe(
    df: pd.DataFrame,
    *,
    run_config: Optional[RunConfig] = None,
    cache: Optional[ResponseCache] = None,
    rate_limiter: Optional[RateLimiter] = None,
    historical_metrics: Optional[Sequence[Mapping[str, Any]]] = None,
    dedupe_config: Optional[DedupeConfig] = None,
    split_config: Optional[SplitConfig] = None,
    client: Optional[OpenAI] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]], Dict[str, Any]]:
    """Run deduplication and classification on ``df``."""

    if run_config is None:
        dedupe_cfg = dedupe_config or DedupeConfig()
        split_cfg = split_config or SplitConfig()
        canonical_df, id_mapping, clusters = deduplicate_dataframe(df.copy(), dedupe_cfg)

        for field in CLASSIFICATION_SCHEMA["properties"].keys():
            if field not in canonical_df.columns:
                canonical_df[field] = ""

        engine = Version2RuleEngine()
        for idx, row in canonical_df.iterrows():
            text = f"{row.get('title', '')}\n\n{row.get('body', '')}".strip()
            result = engine.classify(text)
            for field in CLASSIFICATION_SCHEMA["properties"].keys():
                canonical_df.at[idx, field] = getattr(result, field)

        canonical_df = assign_splits(canonical_df, clusters, split_cfg)
        return canonical_df, id_mapping, clusters, {}

    random.seed(run_config.model.seed)
    structured_log(
        logging.INFO,
        event="dedupe_start",
        enabled=run_config.dedupe.enabled,
        similarity_threshold=run_config.dedupe.similarity_threshold,
        soft_similarity_threshold=run_config.dedupe.soft_similarity_threshold,
    )
    dedupe_start = time.time()
    canonical_df, id_mapping, clusters = deduplicate_dataframe(df, run_config.dedupe)
    dedupe_elapsed = time.time() - dedupe_start
    duplicates_removed = len(df) - len(canonical_df)
    structured_log(
        logging.INFO,
        event="dedupe_summary",
        total=len(df),
        canonical=len(canonical_df),
        clusters=len(clusters),
        duplicates_removed=duplicates_removed,
        elapsed_seconds=dedupe_elapsed,
    )

    if run_config.dedupe.report_path:
        write_dedupe_report(run_config.dedupe.report_path, clusters, canonical_df)
        logger.info("Dedupe report written to %s", run_config.dedupe.report_path)

    for field in CLASSIFICATION_SCHEMA["properties"].keys():
        if field not in canonical_df.columns:
            canonical_df[field] = ""

    processed_ids: set[str] = set()
    if run_config.resume.enabled and run_config.resume.checkpoint_path and run_config.resume.checkpoint_path.exists():
        resume_df = pd.read_csv(run_config.resume.checkpoint_path).fillna("")
        if "canonical_post_id" in resume_df.columns:
            processed_ids = set(resume_df["canonical_post_id"].astype(str))
            for field in CLASSIFICATION_SCHEMA["properties"].keys():
                if field in resume_df.columns:
                    mapping = dict(zip(resume_df["canonical_post_id"].astype(str), resume_df[field].astype(str)))
                    canonical_df[field] = canonical_df["canonical_post_id"].astype(str).map(mapping).fillna(canonical_df[field])
            structured_log(
                logging.INFO,
                event="resume_loaded",
                resume_path=str(run_config.resume.checkpoint_path),
                recovered=len(processed_ids),
            )

    classifier = Version2RuleEngine()
    cache_stats: Dict[str, int] = {"hits": 0, "misses": 0}
    cache = cache or ResponseCache(run_config.cache)
    rate_limiter = rate_limiter or RateLimiter(run_config.parallel.rate_limit)
    client = client or OpenAI()

    base_members = _create_member_callables(
        engine=classifier,
        cache=cache,
        cache_stats=cache_stats,
        rate_limiter=rate_limiter,
        model_cfg=run_config.model,
        client=client,
    )

    def _active_member_callables() -> Dict[str, Any]:
        if not run_config.ensemble.enabled:
            return {"rules": base_members["rules"]}
        enabled_set = set(run_config.ensemble.members) | {"rules"}
        return {name: base_members[name] for name in enabled_set if name in base_members}

    ensemble_members = _active_member_callables()

    latencies: List[float] = []
    metadata_records: List[Dict[str, Any]] = []
    dead_letter_queue: List[Dict[str, Any]] = []
    disagreement_buckets: Dict[str, int] = {"high": 0, "low": 0}

    total = len(canonical_df)
    processed = 0
    start_time = time.time()

    structured_log(
        logging.INFO,
        event="labeling_start",
        total=total,
        resumed=len(processed_ids),
    )

    def worker(index: int, row: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        canonical_id = str(row.get("canonical_post_id") or row.get("post_id") or f"row_{index}")
        if canonical_id in processed_ids:
            payload = {field: row.get(field, "") for field in CLASSIFICATION_SCHEMA["properties"].keys()}
            metadata = {"post_id": canonical_id, "disagreement": 0.0, "latency": 0.0, "member_rationales": {}}
            return payload, metadata

        text = f"{row.get('title', '')}\n\n{row.get('body', '')}".strip()
        attempts = 0
        backoff = 1.0
        while attempts < 3:
            attempts += 1
            start = time.time()
            try:
                payload, metadata = ensemble_classify(
                    text=text,
                    engine=classifier,
                    config=run_config.ensemble,
                    member_callables=ensemble_members,
                    disagreement_stats=None,
                )
                latency = time.time() - start
                metadata.setdefault("field_confidence", {})
                payload = dict(payload)
                payload.setdefault("confidence", float(metadata.get("field_confidence", {}).get("intent", 1.0)))
                for field, schema in CLASSIFICATION_SCHEMA["properties"].items():
                    default_value = 0.0 if schema.get("type") == "number" else ""
                    payload.setdefault(field, default_value)
                validate(instance=payload, schema=CLASSIFICATION_SCHEMA)
                metadata.setdefault("member_rationales", {})
                metadata["member_rationales"] = {
                    key: redact_pii(value or "") for key, value in metadata["member_rationales"].items()
                }
                metadata["post_id"] = canonical_id
                metadata["latency"] = latency
                metadata.setdefault("disagreement", 0.0)
                return payload, metadata
            except ValidationError as exc:
                logger.warning("Schema validation failed on attempt %d: %s", attempts, exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Classification attempt %d failed", attempts)
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        raise RuntimeError("classification_failed")

    for start_idx in range(0, total, run_config.parallel.chunk_size):
        end_idx = min(total, start_idx + run_config.parallel.chunk_size)
        chunk = [(idx, canonical_df.iloc[idx].copy()) for idx in range(start_idx, end_idx)]
        results = parallel_process_batch(chunk, worker, parallel_cfg=run_config.parallel)
        for (idx, _row), result in zip(chunk, results):
            if isinstance(result, Exception):
                dead_letter_queue.append(
                    {
                        "canonical_post_id": canonical_df.at[idx, "canonical_post_id"],
                        "error": str(result),
                    }
                )
                continue
            payload, metadata = result
            for field, value in payload.items():
                canonical_df.at[idx, field] = value
            metadata_records.append(metadata)
            latencies.append(metadata.get("latency", 0.0))
            bucket = "high" if metadata.get("disagreement", 0.0) >= run_config.ensemble.disagreement_threshold else "low"
            disagreement_buckets[bucket] = disagreement_buckets.get(bucket, 0) + 1
        processed += len(chunk)
        elapsed = time.time() - start_time
        remaining = total - processed
        eta = (elapsed / processed) * remaining if processed else 0.0
        structured_log(
            logging.INFO,
            event="progress",
            processed=processed,
            total=total,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        )

    canonical_df = assign_splits(canonical_df, clusters, run_config.split)

    evaluation_metrics = evaluate_against_gold(canonical_df, run_config.evaluation)
    historical_drift = compare_against_history(evaluation_metrics, historical_metrics or []) if evaluation_metrics else {}

    ensemble_summary = summarise_member_agreement(metadata_records)
    high_disagreement_cases = [
        record.get("post_id")
        for record in metadata_records
        if record.get("disagreement", 0.0) >= run_config.ensemble.disagreement_threshold
    ]
    ensemble_summary["high_disagreement_ids"] = high_disagreement_cases
    ensemble_summary["buckets"] = disagreement_buckets

    latencies_sorted = sorted(latencies)
    p95_latency = 0.0
    if latencies_sorted:
        index = max(0, min(len(latencies_sorted) - 1, int(0.95 * len(latencies_sorted)) - 1))
        p95_latency = latencies_sorted[index]

    performance_stats: Dict[str, Any] = {
        "total_rows": len(df),
        "total_time_seconds": time.time() - start_time,
        "avg_latency_seconds": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "p95_latency_seconds": float(p95_latency),
        "cache_enabled": run_config.cache.enabled,
        "cache_hits": cache_stats.get("hits", 0),
        "cache_misses": cache_stats.get("misses", 0),
        "dead_letter_count": len(dead_letter_queue),
    }

    duplicate_stats = {"duplicates_removed": duplicates_removed, "clusters": len(clusters)}

    summary_payload = {
        "duplicate_stats": duplicate_stats,
        "cache_stats": cache_stats,
        "ensemble_summary": ensemble_summary,
        "evaluation_metrics": evaluation_metrics,
        "historical_drift": historical_drift,
        "performance_stats": performance_stats,
        "dead_letter_queue": dead_letter_queue,
    }

    structured_log(
        logging.INFO,
        event="classification_summary",
        total=total,
        labeled=len(metadata_records),
        cached=cache_stats.get("hits", 0),
        dead_letter_count=len(dead_letter_queue),
    )

    return canonical_df, id_mapping, clusters, summary_payload


def run_pipeline(
    *,
    df: pd.DataFrame,
    run_config: RunConfig,
    cache: ResponseCache,
    rate_limiter: RateLimiter,
    historical_payloads: Optional[Iterable[Mapping[str, Any]]] = None,
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """Execute the full classification pipeline and write reports."""

    historical_metrics = [payload.get("evaluation", {}) for payload in historical_payloads or []]

    canonical_df, id_mapping, clusters, summary_payload = classify_dataframe(
        df,
        run_config=run_config,
        cache=cache,
        rate_limiter=rate_limiter,
        historical_metrics=historical_metrics,
        client=client,
    )

    config_payload = _normalise_config(asdict(run_config))

    report_summary = generate_summary_report(
        df=canonical_df,
        duplicate_stats=summary_payload.get("duplicate_stats", {}),
        cache_stats=summary_payload.get("cache_stats", {}),
        ensemble_records=summary_payload.get("ensemble_summary", {}),
        evaluation_metrics=summary_payload.get("evaluation_metrics", {}),
        performance_stats=summary_payload.get("performance_stats", {}),
        config_payload=config_payload,
        report_config=run_config.report,
    )

    return {
        "canonical_df": canonical_df,
        "id_mapping": id_mapping,
        "clusters": clusters,
        "summary_payload": summary_payload,
        "config_payload": config_payload,
        "report_summary": report_summary,
    }


__all__ = ["classify_dataframe", "assign_splits", "run_pipeline"]
