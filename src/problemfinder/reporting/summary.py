"""Summary report generation utilities."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, Mapping

import pandas as pd

from problemfinder.reporting.config import ReportConfig

logger = logging.getLogger(__name__)


def _confidence_interval(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    p = count / total
    z = 1.96
    return z * math.sqrt((p * (1 - p)) / total)


def _label_distribution(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    total = len(df)
    distribution = df[column].value_counts().to_dict()
    return {
        "counts": distribution,
        "total": total,
        "confidence_interval": {
            label: _confidence_interval(count, total) for label, count in distribution.items()
        },
    }


def generate_summary_report(
    *,
    df: pd.DataFrame,
    duplicate_stats: Mapping[str, Any],
    cache_stats: Mapping[str, int],
    ensemble_records: Mapping[str, Any],
    evaluation_metrics: Mapping[str, Any],
    performance_stats: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    report_config: ReportConfig,
) -> Dict[str, Any]:
    """Generate an in-memory summary report and optionally persist it."""

    summary: Dict[str, Any] = {
        "inputs": {
            "total_rows": int(performance_stats.get("total_rows", len(df))),
            "canonical_rows": len(df),
            "duplicates_removed": int(duplicate_stats.get("duplicates_removed", 0)),
        },
        "label_distribution": {
            "intent": _label_distribution(df, "intent"),
            "is_problem": _label_distribution(df, "is_problem"),
            "is_software_solvable": _label_distribution(df, "is_software_solvable"),
            "is_external": _label_distribution(df, "is_external"),
        },
        "cache": dict(cache_stats),
        "ensemble": dict(ensemble_records),
        "evaluation": dict(evaluation_metrics),
        "performance": dict(performance_stats),
        "config": dict(config_payload),
    }

    if report_config.path:
        report_config.path.parent.mkdir(parents=True, exist_ok=True)
        with report_config.path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        logger.info("Summary report written to %s", report_config.path)

    return summary


__all__ = ["generate_summary_report"]
