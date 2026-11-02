
from pathlib import Path

from hallucination_guard.retrieval import EvidenceRetriever


def test_incremental_retrieval():
    project_dir = Path(__file__).resolve().parents[1]
    corpus_path = project_dir / "data" / "sample" / "microsoft.json"

    retriever = EvidenceRetriever(str(corpus_path))

    first_batch = retriever.retrieve(
        "Who founded Microsoft in 1975?",
        top_k=1,
        offset=0,
    )

    second_batch = retriever.retrieve(
        "Who founded Microsoft in 1975?",
        top_k=1,
        offset=1,
    )

    assert len(first_batch) == 1
    assert len(second_batch) == 1
    assert first_batch[0]["id"] != second_batch[0]["id"]


def test_retrieval_respects_maximum_corpus_size():
    project_dir = Path(__file__).resolve().parents[1]
    corpus_path = project_dir / "data" / "sample" / "microsoft.json"

    retriever = EvidenceRetriever(str(corpus_path))

    results = retriever.retrieve(
        "Microsoft founders",
        top_k=10,
    )

    assert len(results) == len(retriever.documents)
