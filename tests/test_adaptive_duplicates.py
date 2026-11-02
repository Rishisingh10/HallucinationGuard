import json

from hallucination_guard.adaptive_verification import (
    AdaptiveVerificationEngine,
)


class MockNLIVerifier:
    def __init__(self):
        self.call_count = 0

    def verify(self, claim, evidence):
        self.call_count += 1

        return {
            "label": "CONTRADICTION",
            "scores": {
                "CONTRADICTION": 0.95,
                "ENTAILMENT": 0.02,
                "NEUTRAL": 0.03,
            },
        }


def test_duplicate_evidence_does_not_inflate_verification(tmp_path):
    corpus = [
        {
            "id": "doc_001",
            "source": "Source A",
            "text": "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        },
        {
            "id": "doc_002",
            "source": "Source B",
            "text": "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        },
        {
            "id": "doc_003",
            "source": "Source C",
            "text": "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        },
    ]

    corpus_path = tmp_path / "duplicates.json"
    corpus_path.write_text(
        json.dumps(corpus),
        encoding="utf-8",
    )

    engine = AdaptiveVerificationEngine(
        corpus_path=str(corpus_path),
        max_passages=3,
    )

    mock_verifier = MockNLIVerifier()
    engine.nli_verifier = mock_verifier

    result = engine.verify(
        "Microsoft was founded by Paul Graham in 1975."
    )

    assert result["status"] == "CONTRADICTED"

    assert result["metrics"]["passages_retrieved"] == 3
    assert result["metrics"]["unique_passages"] == 1
    assert result["metrics"]["duplicate_passages"] == 2

    assert result["metrics"]["nli_evaluations"] == 1
    assert mock_verifier.call_count == 1

    assert result["features"]["contradicting_passages"] == 1