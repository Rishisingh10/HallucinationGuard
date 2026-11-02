
from hallucination_guard.hallucination_guard import HallucinationGuard


def test_guard_integrates_verification_and_correction(tmp_path):
    corpus_path = tmp_path / "empty.json"
    corpus_path.write_text("[]", encoding="utf-8")

    guard = HallucinationGuard(str(corpus_path))

    guard.verification_engine.verify = lambda claim: {
        "claim": claim,
        "status": "SUPPORTED",
        "nli_assessments": [],
    }

    result = guard.analyze("Microsoft was founded in 1975.")

    assert result["claim"] == "Microsoft was founded in 1975."
    assert result["verification"]["status"] == "SUPPORTED"
    assert result["correction"]["action"] == "PRESERVE"


def test_guard_abstains_when_evidence_is_insufficient(tmp_path):
    corpus_path = tmp_path / "empty.json"
    corpus_path.write_text("[]", encoding="utf-8")

    guard = HallucinationGuard(str(corpus_path))

    guard.verification_engine.verify = lambda claim: {
        "claim": claim,
        "status": "INSUFFICIENT",
        "nli_assessments": [],
    }

    result = guard.analyze("Microsoft released Windows in 1975.")

    assert result["verification"]["status"] == "INSUFFICIENT"
    assert result["correction"]["action"] == "ABSTAIN"
    assert result["correction"]["final_claim"] is None
