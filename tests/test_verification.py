
import pytest

from hallucination_guard.verification import EvidenceVerifier


@pytest.fixture
def verifier():
    return EvidenceVerifier()


@pytest.fixture
def microsoft_evidence():
    return [
        {
            "source": "Microsoft History",
            "text": (
                "Microsoft was founded by Bill Gates "
                "and Paul Allen on April 4, 1975."
            ),
        }
    ]


def test_correct_founder_and_year(verifier, microsoft_evidence):
    result = verifier.verify(
        "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        microsoft_evidence,
    )

    assert result["status"] == "SUPPORTED"


def test_incorrect_founder(verifier, microsoft_evidence):
    result = verifier.verify(
        "Microsoft was founded by Bill Gates and Paul Graham in 1975.",
        microsoft_evidence,
    )

    assert result["status"] == "CONTRADICTED"


def test_incorrect_year(verifier, microsoft_evidence):
    result = verifier.verify(
        "Microsoft was founded by Bill Gates and Paul Allen in 1976.",
        microsoft_evidence,
    )

    assert result["status"] == "CONTRADICTED"


def test_incorrect_founder_and_year(verifier, microsoft_evidence):
    result = verifier.verify(
        "Microsoft was founded by Bill Gates and Paul Graham in 1976.",
        microsoft_evidence,
    )

    assert result["status"] == "CONTRADICTED"


def test_no_relevant_evidence(verifier):
    result = verifier.verify(
        "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        [],
    )

    assert result["status"] == "INSUFFICIENT"


def test_founder_mentioned_without_relationship(verifier):
    evidence = [
        {
            "source": "Company Notes",
            "text": "Paul Allen worked at Microsoft during its early years.",
        }
    ]

    result = verifier.verify(
        "Microsoft was founded by Paul Allen in 1975.",
        evidence,
    )

    assert result["status"] == "INSUFFICIENT"


def test_alternative_sentence_structure(verifier):
    evidence = [
        {
            "source": "Microsoft History",
            "text": (
                "Bill Gates and Paul Allen established Microsoft "
                "in Albuquerque, New Mexico, in 1975."
            ),
        }
    ]

    result = verifier.verify(
        "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        evidence,
    )

    assert result["status"] == "SUPPORTED"
