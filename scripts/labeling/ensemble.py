"""Compatibility exports for legacy ensemble imports."""

from labeling.classification.ensemble import (
    PROMPT_VERSION,
    EnsembleConfig,
    EnsembleMemberResult,
    ensemble_classify,
    summarise_member_agreement,
)

__all__ = [
    "PROMPT_VERSION",
    "EnsembleConfig",
    "EnsembleMemberResult",
    "ensemble_classify",
    "summarise_member_agreement",
]
