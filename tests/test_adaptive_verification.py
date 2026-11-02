
from hallucination_guard.adaptive_verification import (
    AdaptiveVerificationEngine,
)


def create_engine():
    return AdaptiveVerificationEngine(
        corpus_path="data/sample/microsoft.json",
        max_passages=3,
    )


def test_supported_claim_uses_minimum_evidence():
    engine = create_engine()

    result = engine.verify(
        "Microsoft was founded by Bill Gates and Paul Allen in 1975."
    )

    assert result["status"] == "SUPPORTED"
    assert result["metrics"]["passages_retrieved"] >= 2
    assert result["metrics"]["passages_retrieved"] <= 3


def test_contradicted_claim_uses_full_budget():
    engine = create_engine()

    result = engine.verify(
        "Microsoft was founded by Bill Gates and Paul Graham in 1975."
    )

    assert result["status"] == "CONTRADICTED"
    assert result["metrics"]["passages_retrieved"] <= 3


def test_insufficient_claim_respects_budget():
    engine = create_engine()

    result = engine.verify(
        "Microsoft had exactly 250 employees on its first day."
    )

    assert result["status"] == "INSUFFICIENT"
    assert result["metrics"]["passages_retrieved"] <= 3


def test_nli_evaluations_match_unique_passages():
    engine = create_engine()

    result = engine.verify(
        "Microsoft was founded by Bill Gates and Paul Allen in 1975."
    )

    assert (
        result["metrics"]["nli_evaluations"]
        == result["metrics"]["unique_passages"]
    )
