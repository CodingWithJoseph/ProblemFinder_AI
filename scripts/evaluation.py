"""Evaluation helpers for the classification pipeline."""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvaluationConfig:
    enabled: bool = False
    gold_set_path: Optional[Path] = None
    metrics: Sequence[str] = dataclasses.field(default_factory=lambda: ["accuracy", "macro_f1", "confusion_matrix"])


def _safe_precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return tp / denom if denom else 0.0


def _safe_recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return tp / denom if denom else 0.0


def _safe_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0


def _confusion_matrix(labels: Sequence[str], predictions: Sequence[str]) -> pd.DataFrame:
    unique_labels = sorted(set(labels) | set(predictions))
    matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels, dtype=int)
    for gold, pred in zip(labels, predictions):
        matrix.loc[gold, pred] += 1
    return matrix


def evaluate_against_gold(predictions: pd.DataFrame, config: EvaluationConfig) -> Dict[str, Any]:
    if not config.enabled or not config.gold_set_path:
        return {}

    gold_path = config.gold_set_path
    if not gold_path.exists():
        raise FileNotFoundError(gold_path)

    gold_df = pd.read_csv(gold_path).fillna("")
    key_column = "canonical_post_id" if "canonical_post_id" in predictions.columns else "post_id"
    if key_column not in gold_df.columns:
        raise ValueError(f"Gold set missing key column {key_column}")

    merged = predictions.merge(gold_df, on=key_column, suffixes=("_pred", "_gold"))
    if merged.empty:
        logger.warning("No overlapping rows between predictions and gold set")
        return {"matches": 0}

    label_columns = [
        ("intent_pred", "intent_gold"),
        ("is_problem_pred", "is_problem_gold"),
        ("is_software_solvable_pred", "is_software_solvable_gold"),
        ("is_external_pred", "is_external_gold"),
    ]

    results: Dict[str, Any] = {"matches": len(merged)}

    for pred_col, gold_col in label_columns:
        if pred_col not in merged.columns or gold_col not in merged.columns:
            continue
        gold = merged[gold_col].astype(str).tolist()
        pred = merged[pred_col].astype(str).tolist()
        correct = sum(1 for g, p in zip(gold, pred) if g == p)
        accuracy = correct / len(gold) if gold else 0.0

        confusion = _confusion_matrix(gold, pred)
        per_label: Dict[str, Dict[str, float]] = {}
        for label in confusion.index:
            tp = confusion.loc[label, label]
            fp = confusion[label].sum() - tp
            fn = confusion.loc[label].sum() - tp
            precision = _safe_precision(tp, fp)
            recall = _safe_recall(tp, fn)
            f1 = _safe_f1(precision, recall)
            per_label[label] = {"precision": precision, "recall": recall, "f1": f1}

        macro_f1 = float(np.mean([stats["f1"] for stats in per_label.values()])) if per_label else 0.0

        results[pred_col.replace("_pred", "")] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_label": per_label,
            "confusion_matrix": confusion.to_dict(),
        }

        if len(gold) >= 30:
            variance = accuracy * (1 - accuracy) / len(gold)
            z = 1.96
            margin = z * np.sqrt(variance)
            results[pred_col.replace("_pred", "")]["confidence_interval"] = (max(0.0, accuracy - margin), min(1.0, accuracy + margin))

    return results


def compare_against_history(current_metrics: Mapping[str, Any], historical_runs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    drifts: Dict[str, Any] = {}
    for metric_name, metric_payload in current_metrics.items():
        if not isinstance(metric_payload, Mapping):
            continue
        historical_values = []
        for historical in historical_runs:
            payload = historical.get(metric_name)
            if isinstance(payload, Mapping) and "accuracy" in payload:
                historical_values.append(payload["accuracy"])
        if not historical_values:
            continue
        baseline = float(np.mean(historical_values))
        current_accuracy = float(metric_payload.get("accuracy", 0.0))
        drift = current_accuracy - baseline
        drifts[metric_name] = {"baseline_accuracy": baseline, "current_accuracy": current_accuracy, "delta": drift}
    return drifts
