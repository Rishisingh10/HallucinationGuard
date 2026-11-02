
import csv
import json
import time
from pathlib import Path
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from hallucination_guard.adaptive_verification import (
    AdaptiveVerificationEngine,
)
from hallucination_guard.evidence_sufficiency import (
    EvidenceSufficiencyGate,
)


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data/scifact/processed/corpus.json"
CLAIMS_PATH = ROOT / "data/scifact/processed/claims_dev.jsonl"
OUTPUT_DIR = ROOT / "data/scifact/evaluation"

LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]
TOP_K = 3


def load_claims():
    with CLAIMS_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize_label(label):
    mapping = {
        "SUPPORT": "SUPPORTED",
        "SUPPORTED": "SUPPORTED",
        "CONTRADICT": "CONTRADICTED",
        "CONTRADICTED": "CONTRADICTED",
        "NOT ENOUGH INFO": "INSUFFICIENT",
        "INSUFFICIENT": "INSUFFICIENT",
        "CONFLICTING": "INSUFFICIENT",
    }

    if label not in mapping:
        raise ValueError(f"Unknown label: {label}")

    return mapping[label]


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, labels=LABELS,
            average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=LABELS,
            average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, labels=LABELS,
            average="macro", zero_division=0
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
    }


def get_gold_document_ids(claim):
    return {
        str(item["doc_id"])
        for item in claim["gold_evidence"]
    }


def get_retrieved_document_ids(documents):
    return {
        str(document["id"]).replace("scifact_", "")
        for document in documents
    }


def retrieval_metrics(results):
    total_relevant = 0
    total_hits = 0
    claims_with_evidence = 0
    claims_with_hit = 0

    for result in results:
        gold_ids = result["gold_document_ids"]

        if not gold_ids:
            continue

        claims_with_evidence += 1
        hits = gold_ids.intersection(
            result["retrieved_document_ids"]
        )

        total_relevant += len(gold_ids)
        total_hits += len(hits)

        if hits:
            claims_with_hit += 1

    return {
        "evidence_recall": (
            total_hits / total_relevant if total_relevant else 0
        ),
        "evidence_hit_rate": (
            claims_with_hit / claims_with_evidence
            if claims_with_evidence else 0
        ),
        "relevant_documents": total_relevant,
        "retrieved_relevant_documents": total_hits,
        "claims_with_annotated_evidence": claims_with_evidence,
        "claims_with_evidence_hit": claims_with_hit,
    }


def run_baseline(engine, claim_text):
    ranking = engine.retriever._bm25_ranking(claim_text)

    documents = [
        engine.retriever.documents[index]
        for index in ranking[:TOP_K]
    ]

    assessments = []

    for document in documents:
        result = engine.nli_verifier.verify(
            claim_text,
            document["text"],
        )

        assessments.append({
            "id": document["id"],
            "source": document["source"],
            "text": document["text"],
            **result,
        })

    if assessments:
        aggregation = engine.aggregator.aggregate(assessments)
        prediction = normalize_label(aggregation["status"])
    else:
        prediction = "INSUFFICIENT"

    return prediction, documents, len(assessments)


def run_full_pipeline(engine, claim_text):
    result = engine.verify(claim_text)

    return (
        normalize_label(result["status"]),
        result["evidence"],
        result["metrics"]["nli_evaluations"],
        result["metrics"]["retrieval_rounds"],
    )


def run_esg_pipeline(engine, gate, claim_text):
    result = engine.verify(claim_text)
    evidence = result["evidence"]

    assessments = []

    for document in evidence:
        assessment = engine.nli_verifier.verify(
            claim_text,
            document["text"],
        )

        assessments.append({
            "id": document["id"],
            **assessment,
        })

    gate_result = gate.evaluate(
        claim_text,
        evidence,
        assessments,
    )

    if gate_result["sufficient"]:
        prediction = normalize_label(result["status"])
    else:
        prediction = "INSUFFICIENT"

    return prediction, gate_result, len(assessments)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    claims = load_claims()
    print(f"Validation claims: {len(claims)}")
    print("Initializing HallucinationGuard...")

    engine = AdaptiveVerificationEngine(
        corpus_path=str(CORPUS_PATH),
        max_passages=TOP_K,
        check_entity_coverage=False,
    )

    gate = EvidenceSufficiencyGate(
        min_claim_coverage=0.25,
        min_nli_confidence=0.75,
    )

    baseline_records = []
    baseline_nli_count = 0

    print("\nStarting BM25 + NLI baseline...")
    baseline_start = time.perf_counter()

    for index, claim in enumerate(claims, start=1):
        gold = normalize_label(claim["label"])

        prediction, documents, nli_count = run_baseline(
            engine, claim["claim"]
        )

        baseline_nli_count += nli_count

        baseline_records.append({
            "id": claim["id"],
            "claim": claim["claim"],
            "gold": gold,
            "prediction": prediction,
            "gold_document_ids": get_gold_document_ids(claim),
            "retrieved_document_ids": get_retrieved_document_ids(
                documents
            ),
        })

        if index % 25 == 0:
            print(f"Baseline: {index}/{len(claims)}")

    baseline_time = time.perf_counter() - baseline_start

    print("\nStarting full HallucinationGuard evaluation...")
    full_start = time.perf_counter()

    results = []
    gold_labels = []
    baseline_predictions = []
    full_predictions = []

    full_nli_count = 0
    full_retrieval_rounds = 0

    for index, (claim, baseline_record) in enumerate(
        zip(claims, baseline_records), start=1
    ):
        prediction, documents, nli_count, rounds = run_full_pipeline(
            engine, claim["claim"]
        )

        full_nli_count += nli_count
        full_retrieval_rounds += rounds

        gold = baseline_record["gold"]

        gold_labels.append(gold)
        baseline_predictions.append(baseline_record["prediction"])
        full_predictions.append(prediction)

        results.append({
            "id": claim["id"],
            "claim": claim["claim"],
            "gold_label": gold,
            "baseline_prediction": baseline_record["prediction"],
            "full_pipeline_prediction": prediction,
            "gold_document_ids": sorted(
                baseline_record["gold_document_ids"]
            ),
            "baseline_retrieved_ids": sorted(
                baseline_record["retrieved_document_ids"]
            ),
            "full_retrieved_ids": sorted(
                get_retrieved_document_ids(documents)
            ),
            "full_nli_evaluations": nli_count,
            "full_retrieval_rounds": rounds,
        })

        if index % 25 == 0:
            print(f"HallucinationGuard: {index}/{len(claims)}")

    full_time = time.perf_counter() - full_start

    print("\nStarting HallucinationGuard + ESG evaluation...")
    esg_start = time.perf_counter()

    esg_predictions = []
    esg_nli_count = 0
    esg_abstentions = 0

    for index, claim in enumerate(claims, start=1):
        prediction, gate_result, nli_count = run_esg_pipeline(
            engine, gate, claim["claim"]
        )

        esg_predictions.append(prediction)
        esg_nli_count += nli_count

        if prediction == "INSUFFICIENT":
            esg_abstentions += 1

        results[index - 1]["esg_prediction"] = prediction
        results[index - 1]["esg_sufficient"] = gate_result["sufficient"]
        results[index - 1]["esg_qualified_passages"] = (
            gate_result["qualified_passages"]
        )
        results[index - 1]["esg_nli_evaluations"] = nli_count

        if index % 25 == 0:
            print(f"HallucinationGuard + ESG: {index}/{len(claims)}")

    esg_time = time.perf_counter() - esg_start

    baseline_metrics = calculate_metrics(
        gold_labels, baseline_predictions
    )

    full_metrics = calculate_metrics(
        gold_labels, full_predictions
    )

    esg_metrics = calculate_metrics(
        gold_labels, esg_predictions
    )

    baseline_retrieval = retrieval_metrics(baseline_records)

    full_retrieval = retrieval_metrics([
        {
            "gold_document_ids": set(item["gold_document_ids"]),
            "retrieved_document_ids": set(item["full_retrieved_ids"]),
        }
        for item in results
    ])

    summary = {
        "dataset": "SciFact validation",
        "total_claims": len(claims),
        "labels": dict(Counter(gold_labels)),
        "baseline_bm25_nli": {
            "classification": baseline_metrics,
            "retrieval": baseline_retrieval,
            "runtime_seconds": baseline_time,
            "nli_evaluations": baseline_nli_count,
        },
        "full_hallucination_guard": {
            "classification": full_metrics,
            "retrieval": full_retrieval,
            "runtime_seconds": full_time,
            "nli_evaluations": full_nli_count,
            "retrieval_rounds": full_retrieval_rounds,
        },
        "hallucination_guard_with_esg": {
            "classification": esg_metrics,
            "runtime_seconds": esg_time,
            "additional_nli_evaluations": esg_nli_count,
            "abstentions": esg_abstentions,
            "abstention_rate": esg_abstentions / len(claims),
            "coverage_rate": 1 - esg_abstentions / len(claims),
            "thresholds": {
                "min_claim_coverage": 0.25,
                "min_nli_confidence": 0.75,
            },
        },
    }

    csv_path = OUTPUT_DIR / "claim_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "evaluation_summary.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n========== CLASSIFICATION RESULTS ==========")

    for name, metrics in [
        ("BASELINE: BM25 + NLI", baseline_metrics),
        ("FULL: HALLUCINATIONGUARD", full_metrics),
        ("HALLUCINATIONGUARD + ESG", esg_metrics),
    ]:
        print(f"\n{name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(metrics["classification_report"])

    print("\n========== RETRIEVAL RESULTS ==========")
    print("Baseline:", baseline_retrieval)
    print("Full pipeline:", full_retrieval)

    print("\n========== PERFORMANCE ==========")
    print(f"Baseline runtime: {baseline_time:.2f} seconds")
    print(f"Full pipeline runtime: {full_time:.2f} seconds")
    print(f"ESG pipeline runtime: {esg_time:.2f} seconds")
    print(f"Baseline NLI evaluations: {baseline_nli_count}")
    print(f"Full NLI evaluations: {full_nli_count}")
    print(f"ESG additional NLI evaluations: {esg_nli_count}")
    print(f"ESG abstentions: {esg_abstentions}/{len(claims)}")

    print("\nGenerated files:")
    print(csv_path)
    print(json_path)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
