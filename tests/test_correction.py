
from hallucination_guard.correction import CorrectionEngine


def nli_result(label, score=0.98):
    scores = {
        "ENTAILMENT": 0.01,
        "CONTRADICTION": 0.01,
        "NEUTRAL": 0.01,
    }
    scores[label] = score

    return {
        "label": label,
        "scores": scores,
    }


class MockNLIVerifier:
    def __init__(self, results):
        self.results = results
        self.call_count = 0

    def verify(self, claim, evidence):
        if self.call_count < len(self.results):
            result = self.results[self.call_count]
        else:
            result = nli_result("NEUTRAL")

        self.call_count += 1
        return result


def make_verification(status="CONTRADICTED"):
    return {
        "claim": "Microsoft was founded by Paul Graham.",
        "status": status,
        "evidence": [
            {
                "id": "doc_001",
                "source": "Source A",
                "text": "Microsoft was founded by Bill Gates and Paul Allen.",
                "score": 5.0,
            },
            {
                "id": "doc_002",
                "source": "Source B",
                "text": "Bill Gates and Paul Allen established Microsoft in 1975.",
                "score": 4.0,
            },
        ],
        "nli_assessments": [
            {
                "id": "doc_001",
                "label": "CONTRADICTION",
                "scores": {"CONTRADICTION": 0.98},
            },
            {
                "id": "doc_002",
                "label": "CONTRADICTION",
                "scores": {"CONTRADICTION": 0.95},
            },
        ],
    }


def test_supported_claim_is_preserved():
    engine = CorrectionEngine()

    result = engine.correct({
        "claim": "Microsoft was founded in 1975.",
        "status": "SUPPORTED",
    })

    assert result["action"] == "PRESERVE"
    assert result["final_claim"] == "Microsoft was founded in 1975."


def test_correction_requires_second_entailment_check():
    engine = CorrectionEngine()

    verifier = MockNLIVerifier([
        nli_result("ENTAILMENT"),
    ])

    result = engine.correct(
        make_verification(),
        verifier,
    )

    assert result["action"] == "REPLACE_WITH_EVIDENCE"
    assert result["evidence_id"] == "doc_001"
    assert result["validation_count"] == 1


def test_correction_abstains_without_second_verification():
    engine = CorrectionEngine()

    result = engine.correct(make_verification())

    assert result["action"] == "ABSTAIN"
    assert result["final_claim"] is None


def test_unsupported_candidate_causes_abstention():
    engine = CorrectionEngine()

    verifier = MockNLIVerifier([
        nli_result("NEUTRAL"),
    ])

    result = engine.correct(make_verification(), verifier)

    assert result["action"] == "ABSTAIN"


def test_conflicting_candidate_validation_causes_abstention():
    engine = CorrectionEngine()

    verifier = MockNLIVerifier([
        nli_result("ENTAILMENT"),
        nli_result("CONTRADICTION"),
    ])

    verification = make_verification()

    verification["evidence"].append({
        "id": "doc_003",
        "source": "Source C",
        "text": "Microsoft was established by Bill Gates in 1976.",
        "score": 3.0,
    })

    result = engine.correct(verification, verifier)

    assert result["action"] == "ABSTAIN"


def test_weak_candidate_support_causes_abstention():
    engine = CorrectionEngine()

    verifier = MockNLIVerifier([
        nli_result("ENTAILMENT", 0.75),
    ])

    result = engine.correct(make_verification(), verifier)

    assert result["action"] == "ABSTAIN"


def test_insufficient_evidence_abstains():
    engine = CorrectionEngine()

    result = engine.correct({
        "claim": "Microsoft released Windows in 1975.",
        "status": "INSUFFICIENT",
    })

    assert result["action"] == "ABSTAIN"


def test_conflicting_original_evidence_abstains():
    engine = CorrectionEngine()

    result = engine.correct({
        "claim": "Microsoft was founded in 1975.",
        "status": "CONFLICTING",
    })

    assert result["action"] == "ABSTAIN"


def test_missing_candidate_evidence_abstains():
    engine = CorrectionEngine()

    verification = make_verification()
    verification["evidence"] = []

    result = engine.correct(
        verification,
        MockNLIVerifier([]),
    )

    assert result["action"] == "ABSTAIN"
