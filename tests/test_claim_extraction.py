
from hallucination_guard.claim_extraction import ClaimExtractor


def test_extract_multiple_claims():
    extractor = ClaimExtractor()

    response = (
        "Microsoft was founded in 1975. "
        "Bill Gates and Paul Allen founded it."
    )

    claims = extractor.extract(response)

    assert len(claims) == 2
    assert "Microsoft was founded in 1975." == claims[0]
    assert "Bill Gates and Paul Allen founded it." == claims[1]


def test_extract_empty_response():
    extractor = ClaimExtractor()

    assert extractor.extract("") == []
