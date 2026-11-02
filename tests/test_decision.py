
from hallucination_guard.decision import VerificationDecisionEngine


def test_supported_claim():
    engine = VerificationDecisionEngine()
    features = {"evidence_uncertainty": 0.02}

    result = engine.decide(features, "SUPPORTED")

    assert result["risk_level"] == "LOW"
    assert result["action"] == "ACCEPT_PROVISIONALLY"


def test_contradicted_claim():
    engine = VerificationDecisionEngine()
    features = {"evidence_uncertainty": 0.02}

    result = engine.decide(features, "CONTRADICTED")

    assert result["risk_level"] == "HIGH"
    assert result["action"] == "FLAG"


def test_conflicting_evidence():
    engine = VerificationDecisionEngine()
    features = {"evidence_uncertainty": 0.02}

    result = engine.decide(features, "CONFLICTING")

    assert result["risk_level"] == "HIGH"
    assert result["action"] == "INVESTIGATE"


def test_insufficient_evidence():
    engine = VerificationDecisionEngine()
    features = {"evidence_uncertainty": None}

    result = engine.decide(features, "INSUFFICIENT")

    assert result["risk_level"] == "UNKNOWN"
    assert result["action"] == "RETRIEVE_MORE"


def test_uncertain_nli_predictions():
    engine = VerificationDecisionEngine()
    features = {"evidence_uncertainty": 0.75}

    result = engine.decide(features, "SUPPORTED")

    assert result["risk_level"] == "MEDIUM"
    assert result["action"] == "RETRIEVE_MORE"
