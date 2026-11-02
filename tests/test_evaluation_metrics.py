
import pytest

from hallucination_guard.evaluation_metrics import EvaluationMetrics


@pytest.fixture
def evaluator():
    return EvaluationMetrics()


def test_perfect_predictions(evaluator):
    labels = [
        "SUPPORTED",
        "CONTRADICTED",
        "INSUFFICIENT",
        "CONFLICTING",
    ]

    result = evaluator.evaluate(labels, labels)

    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_incorrect_predictions(evaluator):
    actual = ["SUPPORTED", "CONTRADICTED"]
    predicted = ["CONTRADICTED", "SUPPORTED"]

    result = evaluator.evaluate(actual, predicted)

    assert result["accuracy"] == 0.0
    assert result["total_samples"] == 2


def test_confusion_matrix_shape(evaluator):
    actual = ["SUPPORTED", "CONTRADICTED"]
    predicted = ["SUPPORTED", "INSUFFICIENT"]

    result = evaluator.evaluate(actual, predicted)

    assert len(result["confusion_matrix"]) == 4
    assert all(
        len(row) == 4
        for row in result["confusion_matrix"]
    )


def test_per_class_metrics(evaluator):
    actual = ["SUPPORTED", "SUPPORTED", "CONTRADICTED"]
    predicted = ["SUPPORTED", "CONTRADICTED", "CONTRADICTED"]

    result = evaluator.evaluate(actual, predicted)

    assert result["per_class"]["SUPPORTED"]["support"] == 2
    assert result["per_class"]["CONTRADICTED"]["support"] == 1


def test_empty_evaluation_data(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate([], [])


def test_mismatched_lengths(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate(
            ["SUPPORTED"],
            ["SUPPORTED", "CONTRADICTED"],
        )


def test_invalid_labels(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate(
            ["UNKNOWN"],
            ["SUPPORTED"],
        )
