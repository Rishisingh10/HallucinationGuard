
from hallucination_guard.evidence_sufficiency import EvidenceSufficiencyGate

gate = EvidenceSufficiencyGate(
    min_claim_coverage=0.25,
    min_nli_confidence=0.75,
)

claim = "Microsoft was founded by Bill Gates and Paul Allen in 1975."

# Test 1: Relevant evidence with strong entailment
evidence_1 = [
    {
        "id": "doc1",
        "text": "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
    }
]

assessments_1 = [
    {
        "id": "doc1",
        "label": "ENTAILMENT",
        "scores": {
            "ENTAILMENT": 0.96,
            "CONTRADICTION": 0.01,
            "NEUTRAL": 0.03,
        },
    }
]

# Test 2: Irrelevant evidence with strong contradiction
evidence_2 = [
    {
        "id": "doc2",
        "text": "Microsoft develops software and cloud computing products.",
    }
]

assessments_2 = [
    {
        "id": "doc2",
        "label": "CONTRADICTION",
        "scores": {
            "ENTAILMENT": 0.01,
            "CONTRADICTION": 0.96,
            "NEUTRAL": 0.03,
        },
    }
]

for name, evidence, assessments, expected in [
    ("Relevant evidence", evidence_1, assessments_1, True),
    ("Irrelevant evidence", evidence_2, assessments_2, False),
]:
    result = gate.evaluate(claim, evidence, assessments)

    print(f"\n{name}")
    print("Expected:", expected)
    print("Actual:", result["sufficient"])
    print("Details:", result["passages"])

    assert result["sufficient"] == expected

print("\nBoth corrected gate tests passed.")
