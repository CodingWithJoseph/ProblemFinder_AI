from labeling.classification import (
    EnsembleConfig,
    EnsembleMemberResult,
    ensemble_classify,
)
from labeling.classification import Version2RuleEngine


def test_ensemble_majority_vote_prefers_confident_member():
    engine = Version2RuleEngine()
    config = EnsembleConfig(enabled=True, members=["rules", "direct", "reasoning"], disagreement_threshold=0.5)

    def high_confidence_member(*, text: str):
        payload = engine.classify(text)
        return EnsembleMemberResult(
            member="direct",
            payload={field: getattr(payload, field) for field in payload.__dataclass_fields__},
            confidence=0.9,
            rationale="LLM rationale",
        )

    def low_confidence_member(*, text: str):
        payload = engine.classify(text)
        payload.is_external = "0"
        return EnsembleMemberResult(
            member="reasoning",
            payload={field: getattr(payload, field) for field in payload.__dataclass_fields__},
            confidence=0.1,
            rationale="Second opinion",
        )

    results, metadata = ensemble_classify(
        text="Need help deciding if I should buy a new laptop",
        engine=engine,
        config=config,
        member_callables={
            "rules": lambda text: None,
            "direct": high_confidence_member,
            "reasoning": low_confidence_member,
        },
        disagreement_stats={},
    )

    assert results["is_external"] == "1"
    assert metadata["member_rationales"].keys() == {"direct", "reasoning", "rules"}
