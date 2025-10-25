"""Ensemble aggregation utilities for combining model and rule outputs."""

from __future__ import annotations

import dataclasses
import logging
import statistics
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from labeling.classification.rules import Version2RuleEngine

logger = logging.getLogger(__name__)

PROMPT_VERSION = "2.0.0"


@dataclasses.dataclass(slots=True)
class EnsembleConfig:
    """Configuration controlling ensemble behaviour."""

    enabled: bool = False
    members: Sequence[str] = dataclasses.field(default_factory=lambda: ["direct", "reasoning", "rules"])
    disagreement_threshold: float = 0.3


@dataclasses.dataclass(slots=True)
class EnsembleMemberResult:
    """Result produced by an ensemble member."""

    member: str
    payload: Mapping[str, str]
    confidence: float
    rationale: str


def _majority_vote(
    *,
    field: str,
    member_results: Sequence[EnsembleMemberResult],
    priorities: Mapping[str, int],
) -> Tuple[str, float]:
    """Return selected label and aggregated confidence for ``field``."""

    weighted_scores: MutableMapping[str, float] = {}
    for result in member_results:
        value = result.payload.get(field, "")
        if not value:
            continue
        weighted_scores[value] = weighted_scores.get(value, 0.0) + max(result.confidence, 0.0)

    if not weighted_scores:
        return "", 0.0

    best_value, best_score = max(weighted_scores.items(), key=lambda item: item[1])

    tied_values = [value for value, score in weighted_scores.items() if score == best_score]
    if len(tied_values) == 1:
        return best_value, best_score

    def tie_key(value: str) -> Tuple[int, str]:
        best_priority = min(
            (priorities.get(res.member, 1000) for res in member_results if res.payload.get(field, "") == value),
            default=1000,
        )
        return best_priority, value

    chosen = min(tied_values, key=tie_key)
    return chosen, best_score


def _calculate_disagreement(member_results: Sequence[EnsembleMemberResult]) -> float:
    """Return the proportion of fields where members disagreed."""

    if not member_results:
        return 0.0
    disagreements = 0
    total = 0
    for field in member_results[0].payload.keys():
        labels = [result.payload.get(field, "") for result in member_results]
        if not labels:
            continue
        total += 1
        if len(set(labels)) > 1:
            disagreements += 1
    if total == 0:
        return 0.0
    return disagreements / total


def rule_based_classifier(text: str, engine: Version2RuleEngine) -> EnsembleMemberResult:
    """Thin wrapper around the rule engine."""

    classification = engine.classify(text)
    payload = {field: getattr(classification, field) for field in classification.__dataclass_fields__}
    rationale = classification.problem_reason or classification.software_reason or classification.external_reason
    confidence = float(classification.confidence or 1.0)
    return EnsembleMemberResult(member="rules", payload=payload, confidence=confidence, rationale=rationale)


def ensemble_classify(
    *,
    text: str,
    engine: Version2RuleEngine,
    config: EnsembleConfig,
    member_callables: Mapping[str, Any],
    disagreement_stats: Optional[MutableMapping[str, int]] = None,
) -> Tuple[Mapping[str, str], Dict[str, Any]]:
    """Run the reasoning ensemble and return the selected payload and metadata."""

    if not config.enabled:
        result = rule_based_classifier(text, engine)
        metadata = {
            "member_rationales": {result.member: result.rationale},
            "member_confidence": {result.member: result.confidence},
        }
        return result.payload, metadata

    active_members = [member for member in config.members if member in member_callables]
    priorities = {"reasoning": 0, "direct": 1, "rules": 2}

    member_results: List[EnsembleMemberResult] = []
    member_details: Dict[str, Dict[str, Any]] = {}

    for member in active_members:
        try:
            callable_obj = member_callables[member]
            if member == "rules":
                result = rule_based_classifier(text, engine)
            else:
                result = callable_obj(text=text)
            member_results.append(result)
            member_details[member] = {
                "confidence": result.confidence,
                "rationale": result.rationale,
                "payload": dict(result.payload),
            }
        except Exception:  # pragma: no cover - defensive
            logger.exception("Ensemble member %s failed", member)

    if not member_results:
        fallback = rule_based_classifier(text, engine)
        return fallback.payload, {"member_rationales": {fallback.member: fallback.rationale}}

    fields = list(member_results[0].payload.keys())
    aggregated: Dict[str, str] = {}
    confidences: Dict[str, float] = {}

    for field in fields:
        value, confidence = _majority_vote(field=field, member_results=member_results, priorities=priorities)
        aggregated[field] = value
        confidences[field] = confidence

    disagreement = _calculate_disagreement(member_results)
    if disagreement_stats is not None:
        bucket = "high" if disagreement >= config.disagreement_threshold else "low"
        disagreement_stats[bucket] = disagreement_stats.get(bucket, 0) + 1

    if disagreement >= config.disagreement_threshold:
        logger.warning("High ensemble disagreement %.2f for text hash=%s", disagreement, hash(text))

    metadata = {
        "member_rationales": {member: details.get("rationale") for member, details in member_details.items()},
        "member_confidence": {member: details.get("confidence") for member, details in member_details.items()},
        "field_confidence": confidences,
        "disagreement": disagreement,
    }
    return aggregated, metadata


def summarise_member_agreement(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Return mean/stdev disagreement metrics for monitoring."""

    disagreements = [record.get("disagreement", 0.0) for record in records]
    if not disagreements:
        return {"mean": 0.0, "stdev": 0.0, "count": 0}
    return {
        "mean": statistics.fmean(disagreements),
        "stdev": statistics.pstdev(disagreements) if len(disagreements) > 1 else 0.0,
        "count": len(disagreements),
    }
