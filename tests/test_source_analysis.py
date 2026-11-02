
from hallucination_guard.source_analysis import SourceAnalyzer


def test_unique_sources():
    analyzer = SourceAnalyzer()

    evidence = [
        {
            "id": "doc_001",
            "source": "Source A",
            "text": "Microsoft was founded in 1975.",
        },
        {
            "id": "doc_002",
            "source": "Source B",
            "text": "Bill Gates and Paul Allen established Microsoft.",
        },
    ]

    result = analyzer.analyze(evidence)

    assert result["unique_source_count"] == 2
    assert result["unique_passages"] == 2


def test_duplicate_passages():
    analyzer = SourceAnalyzer()

    evidence = [
        {
            "id": "doc_001",
            "source": "Source A",
            "text": "Microsoft was founded in 1975.",
        },
        {
            "id": "doc_002",
            "source": "Source B",
            "text": "Microsoft was founded in 1975.",
        },
    ]

    result = analyzer.analyze(evidence)

    assert result["duplicate_passages"] == 1
    assert result["unique_passages"] == 1


def test_same_source_multiple_passages():
    analyzer = SourceAnalyzer()

    evidence = [
        {
            "id": "doc_001",
            "source": "Source A",
            "text": "Microsoft was founded in 1975.",
        },
        {
            "id": "doc_002",
            "source": "Source A",
            "text": "Bill Gates and Paul Allen founded Microsoft.",
        },
    ]

    result = analyzer.analyze(evidence)

    assert result["unique_source_count"] == 1
    assert result["unique_passages"] == 2


def test_empty_evidence():
    analyzer = SourceAnalyzer()

    result = analyzer.analyze([])

    assert result["retrieved_passages"] == 0
    assert result["unique_sources_count"] == 0 if "unique_sources_count" in result else result["unique_source_count"] == 0
    assert result["duplicate_passages"] == 0
