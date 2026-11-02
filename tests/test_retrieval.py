
from pathlib import Path

from hallucination_guard.retrieval import EvidenceRetriever


def test_retrieval():
    corpus_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample"
        / "microsoft.json"
    )

    retriever = EvidenceRetriever(str(corpus_path))

    results = retriever.retrieve(
        "Who founded Microsoft?",
        top_k=3,
    )

    assert len(results) == 3
    assert "Paul Allen" in results[0]["text"]
