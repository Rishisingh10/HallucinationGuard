
import json
from pathlib import Path

from hallucination_guard.retrieval import EvidenceRetriever
from hallucination_guard.nli_verification import NLIVerifier
from hallucination_guard.evidence_aggregation import EvidenceAggregator
from hallucination_guard.adaptive_verification import AdaptiveVerificationEngine


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data/scifact/processed/corpus.json"
CLAIMS_PATH = ROOT / "data/scifact/processed/claims_dev.jsonl"


def main():
    with CLAIMS_PATH.open(encoding="utf-8") as file:
        claim = json.loads(next(file))

    print("\nCLAIM")
    print(claim["claim"])
    print("GOLD LABEL:", claim["label"])

    print("\nINITIALIZING RETRIEVER...")
    retriever = EvidenceRetriever(str(CORPUS_PATH))

    documents = retriever.retrieve(claim["claim"], top_k=3)

    print("\nRETRIEVED DOCUMENTS:", len(documents))

    verifier = NLIVerifier()
    assessments = []

    for document in documents:
        result = verifier.verify(
            claim["claim"],
            document["text"],
        )

        assessments.append({
            "id": document["id"],
            "source": document["source"],
            "text": document["text"],
            **result,
        })

        print("\nNLI ASSESSMENT")
        print(json.dumps(result, indent=2))

    aggregator = EvidenceAggregator()
    aggregation = aggregator.aggregate(assessments)

    print("\nBASELINE AGGREGATION")
    print(json.dumps(aggregation, indent=2))

    print("\nFULL HALLUCINATIONGUARD PIPELINE")

    engine = AdaptiveVerificationEngine(
        corpus_path=str(CORPUS_PATH),
        max_passages=3,
        check_entity_coverage=False,
    )

    result = engine.verify(claim["claim"])

    print("STATUS:", result["status"])
    print("METRICS:", json.dumps(result["metrics"], indent=2))
    print("STOP REASON:", result["stop_reason"])

    print("\nPIPELINE SMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()
