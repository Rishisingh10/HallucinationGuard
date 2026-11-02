
import pytest

from hallucination_guard.verification_budget import VerificationBudget


def test_initial_budget():
    budget = VerificationBudget()

    assert budget.initial_budget() == 1


def test_stop_for_strong_support():
    budget = VerificationBudget()

    features = {
        "evidence_uncertainty": 0.02,
        "evidence_conflict": False,
    }

    first_result = budget.next_batch(
        "SUPPORTED",
        features,
        retrieved_count=1,
    )

    assert first_result["retrieve"] is True
    assert first_result["count"] == 1

    second_result = budget.next_batch(
        "SUPPORTED",
        features,
        retrieved_count=2,
    )

    assert second_result["retrieve"] is False


def test_continue_for_high_uncertainty():
    budget = VerificationBudget()

    features = {
        "evidence_uncertainty": 0.75,
        "evidence_conflict": False,
    }

    result = budget.next_batch(
        "SUPPORTED",
        features,
        retrieved_count=2,
    )

    assert result["retrieve"] is True


def test_continue_for_conflicting_evidence():
    budget = VerificationBudget()

    features = {
        "evidence_uncertainty": 0.20,
        "evidence_conflict": True,
    }

    result = budget.next_batch(
        "CONFLICTING",
        features,
        retrieved_count=2,
    )

    assert result["retrieve"] is True


def test_continue_for_insufficient_evidence():
    budget = VerificationBudget()

    features = {
        "evidence_uncertainty": None,
        "evidence_conflict": False,
    }

    result = budget.next_batch(
        "INSUFFICIENT",
        features,
        retrieved_count=2,
    )

    assert result["retrieve"] is True


def test_stop_at_maximum_budget():
    budget = VerificationBudget(max_passages=3)

    result = budget.next_batch(
        "CONTRADICTED",
        {},
        retrieved_count=3,
    )

    assert result["retrieve"] is False
    assert result["count"] == 0


def test_contradiction_gets_additional_evidence():
    budget = VerificationBudget()

    result = budget.next_batch(
        "CONTRADICTED",
        {
            "evidence_uncertainty": 0.05,
            "evidence_conflict": False,
        },
        retrieved_count=2,
    )

    assert result["retrieve"] is True


def test_invalid_budget_configuration():
    with pytest.raises(ValueError):
        VerificationBudget(initial_passages=0)

    with pytest.raises(ValueError):
        VerificationBudget(
            initial_passages=2,
            max_passages=1,
        )

    with pytest.raises(ValueError):
        VerificationBudget(batch_size=0)

    with pytest.raises(ValueError):
        VerificationBudget(
            max_passages=3,
            min_passages_before_stop=4,
        )
