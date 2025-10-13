"""Compatibility exports for legacy evaluation imports."""

from problemfinder.reporting.evaluation import (
    EvaluationConfig,
    compare_against_history,
    evaluate_against_gold,
)

__all__ = ["EvaluationConfig", "compare_against_history", "evaluate_against_gold"]
