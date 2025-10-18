"""Configuration loading utilities for pipeline runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from problemfinder.classification.ensemble import EnsembleConfig
from problemfinder.core.cache import CacheConfig
from problemfinder.core.config import (
    DedupeConfig,
    ModelConfig,
    ParallelConfig,
    ResumeConfig,
    RunConfig,
    SplitConfig,
)
from problemfinder.reporting.config import ReportConfig
from problemfinder.reporting.evaluation import EvaluationConfig


def _load_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    """
    - Opens and safely parses a YAML file into a Python dictionary.
    - Returns an empty dict if the file is missing.
    - Performs type-checking to ensure the YAML contains a valid dictionary structure.
    - Used internally by `build_run_config`.
    """
    if path and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid YAML config structure at {path}")
        return payload
    return {}


def build_run_config(args: argparse.Namespace) -> RunConfig:
    # Load the YAML configuration and overlay CLI overrides.
    config_path = Path(args.config) if getattr(args, "config", None) else None
    yaml_payload = _load_yaml_config(config_path)

    # Model Configuration
    model_cfg = ModelConfig(
        name=str(getattr(args, "model", yaml_payload.get("model", "gpt-4o"))),
        temperature=float(yaml_payload.get("temperature", 0.0)),
        seed=int(yaml_payload.get("seed", 42)),
    )
    if getattr(args, "temperature", None) is not None:
        model_cfg.temperature = float(args.temperature)
    if getattr(args, "seed", None) is not None:
        model_cfg.seed = int(args.seed)

    # Ensemble Configuration
    ensemble_section = yaml_payload.get("ensemble", {}) if isinstance(yaml_payload, dict) else {}
    ensemble_cfg = EnsembleConfig(
        enabled=bool(ensemble_section.get("enabled", False)),
        members=ensemble_section.get("members", ["direct", "reasoning", "rules"]),
        disagreement_threshold=float(ensemble_section.get("disagreement_threshold", 0.3)),
    )
    if getattr(args, "ensemble", None) is not None:
        ensemble_cfg.enabled = args.ensemble == "on"
    if getattr(args, "ensemble_members", None):
        ensemble_cfg.members = [member.strip() for member in args.ensemble_members.split(",") if member.strip()]
    if getattr(args, "ensemble_disagreement_threshold", None) is not None:
        ensemble_cfg.disagreement_threshold = float(args.ensemble_disagreement_threshold)

    # Cache Configuration
    cache_section = yaml_payload.get("cache", {}) if isinstance(yaml_payload, dict) else {}
    cache_cfg = CacheConfig(
        enabled=bool(cache_section.get("enabled", False)),
        path=Path(cache_section.get("path", "data/.cache/responses.json")),
        ttl_hours=int(cache_section.get("ttl_hours", 24)),
    )
    if getattr(args, "cache", None) is not None:
        cache_cfg.enabled = args.cache == "on"
    if getattr(args, "cache_ttl", None) is not None:
        cache_cfg.ttl_hours = int(args.cache_ttl)
    if getattr(args, "cache_path", None):
        cache_cfg.path = Path(args.cache_path)

    # Parallel Configuration
    parallel_section = yaml_payload.get("parallel", {}) if isinstance(yaml_payload, dict) else {}
    parallel_cfg = ParallelConfig(
        max_workers=int(parallel_section.get("max_workers", 4)),
        rate_limit=int(parallel_section.get("rate_limit", 30)),
        chunk_size=int(parallel_section.get("chunk_size", 50)),
    )
    if getattr(args, "max_workers", None) is not None:
        parallel_cfg.max_workers = max(1, int(args.max_workers))
    if getattr(args, "rate_limit", None) is not None:
        parallel_cfg.rate_limit = max(0, int(args.rate_limit))
    if getattr(args, "chunk_size", None) is not None:
        parallel_cfg.chunk_size = max(1, int(args.chunk_size))

    # Evaluation Configuration
    evaluation_section = yaml_payload.get("evaluation", {}) if isinstance(yaml_payload, dict) else {}
    evaluation_cfg = EvaluationConfig(
        enabled=bool(evaluation_section.get("enabled", False)),
        gold_set_path=Path(evaluation_section.get("gold_set_path")) if evaluation_section.get("gold_set_path") else None,
        metrics=evaluation_section.get("metrics", ["accuracy", "macro_f1", "confusion_matrix"]),
    )
    if getattr(args, "evaluation", None) is not None:
        evaluation_cfg.enabled = args.evaluation == "on"
    if getattr(args, "evaluation_gold_set", None):
        evaluation_cfg.gold_set_path = Path(args.evaluation_gold_set)

    # Reporting Configuration
    report_section = yaml_payload.get("report", {}) if isinstance(yaml_payload, dict) else {}
    report_cfg = ReportConfig(path=Path(report_section.get("path")) if report_section.get("path") else None)
    if getattr(args, "report_path", None):
        report_cfg.path = Path(args.report_path)

    # Deduplication Configuration
    cross_subreddit_setting = yaml_payload.get("cross_subreddit", True)
    if getattr(args, "cross_subreddit_dedupe", False):
        cross_subreddit_setting = True

    dedupe_cfg = DedupeConfig(
        enabled=(args.dedupe == "on"),
        similarity_threshold=float(args.similarity_threshold),
        canonical_policy=args.canonical_policy,
        report_path=args.dedupe_report,
        soft_similarity_threshold=float(
            getattr(args, "soft_similarity_threshold", yaml_payload.get("soft_similarity_threshold", 0.35))
        ),
        cross_subreddit=bool(cross_subreddit_setting),
    )
    if isinstance(dedupe_cfg.report_path, str):
        dedupe_cfg.report_path = Path(dedupe_cfg.report_path)

    # Split Configuration
    split_cfg = SplitConfig(
        enabled=not args.no_split,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )

    # Resume Configuration
    resume_cfg = ResumeConfig(
        enabled=bool(getattr(args, "resume", False)),
        checkpoint_path=Path(args.resume_from) if getattr(args, "resume_from", None) else None,
    )

    # Aggregate all configurations into a single RunConfig object.
    return RunConfig(
        model=model_cfg,
        ensemble=ensemble_cfg,
        cache=cache_cfg,
        parallel=parallel_cfg,
        evaluation=evaluation_cfg,
        report=report_cfg,
        dedupe=dedupe_cfg,
        split=split_cfg,
        resume=resume_cfg,
    )
