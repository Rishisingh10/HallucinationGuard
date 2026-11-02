
from pathlib import Path

from hallucination_guard.retrieval import EvidenceRetriever


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

for result in results:
    print({
        "source": result["source"],
        "score": round(result["score"], 4),
        "text": result["text"],
    })
