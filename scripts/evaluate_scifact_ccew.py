
import json
import time
from pathlib import Path

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
from hallucination_guard.sentence_evidence_selector import (
    SentenceEvidenceSelector,
)


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data/scifact/processed/corpus.json"
CLAIMS_PATH = ROOT / "data/scifact/processed/claims_dev.jsonl"
OUTPUT_DIR = ROOT / "data/scifact/evaluation"

LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]
TOP_K = 3


def load_claims():
    with CLAIMS_PATH.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


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
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        ).tolist(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    claims = load_claims()
    print(f"SciFact claims: {len(claims)}")

    print("Initializing HallucinationGuard...")

    engine = AdaptiveVerificationEngine(
        corpus_path=str(CORPUS_PATH),
        max_passages=TOP_K,
        check_entity_coverage=False,
    )

    selector = SentenceEvidenceSelector(window_size=2)

    y_true = []
    baseline_predictions = []
    ccew_predictions = []

    claim_results = []

    baseline_nli_count = 0
    ccew_nli_count = 0

    baseline_start = time.perf_counter()

    print("\nRunning original full pipeline...")

    for index, claim in enumerate(claims, start=1):
        result = engine.verify(claim["claim"])

        baseline_prediction = normalize_label(result["status"])

        evidence = result["evidence"]

        baseline_nli_count += result["metrics"]["nli_evaluations"]

        # Select focused sentence windows from the SAME
        # retrieved evidence. Do not perform retrieval again.

        selected_evidence = selector.select(
            claim["claim"],
            evidence,
        )

        assessments = []

        for document in selected_evidence:
            assessment = engine.nli_verifier.verify(
                claim["claim"],
                document["text"],
            )

            ccew_nli_count += 1

            assessments.append({
                "id": document["id"],
                "source": document.get("source"),
                "text": document["text"],
                **assessment,
            })

        if assessments:
            aggregation = engine.aggregator.aggregate(
                assessments
            )
            ccew_prediction = normalize_label(
                aggregation["status"]
            )
        else:
            ccew_prediction = "INSUFFICIENT"

        gold = normalize_label(claim["label"])

        y_true.append(gold)
        baseline_predictions.append(baseline_prediction)
        ccew_predictions.append(ccew_prediction)

        claim_results.append({
            "claim_id": claim["id"],
            "gold_label": gold,
            "baseline_prediction": baseline_prediction,
            "ccew_prediction": ccew_prediction,
            "retrieved_passages": len(evidence),
            "selected_passages": len(selected_evidence),
            "baseline_nli_evaluations": (
                result["metrics"]["nli_evaluations"]
            ),
            "ccew_nli_evaluations": len(assessments),
        })

        if index % 25 == 0 or index == len(claims):
            print(f"Processed {index}/{len(claims)} claims")

    runtime = time.perf_counter() - baseline_start

    baseline_metrics = calculate_metrics(
        y_true,
        baseline_predictions,
    )

    ccew_metrics = calculate_metrics(
        y_true,
        ccew_predictions,
    )

    summary = {
        "dataset": "SciFact development set",
        "total_claims": len(claims),
        "experiment": (
            "Same adaptive retrieval; original document text "
            "versus claim-conditioned sentence windows."
        ),
        "baseline": {
            "classification": baseline_metrics,
            "nli_evaluations": baseline_nli_count,
        },
        "ccew": {
            "classification": ccew_metrics,
            "nli_evaluations": ccew_nli_count,
            "window_size": 2,
        },
        "runtime_seconds": runtime,
    }

    summary_path = OUTPUT_DIR / "ccew_evaluation_summary.json"
    claims_path = OUTPUT_DIR / "ccew_claim_results.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with claims_path.open("w", encoding="utf-8") as file:
        json.dump(claim_results, file, indent=2)

    print("\n========== CCEW RESULTS ==========")

    for name, metrics in [
        ("ORIGINAL FULL PIPELINE", baseline_metrics),
        ("HALLUCINATIONGUARD + CCEW", ccew_metrics),
    ]:
        print(f"\n{name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(metrics["classification_report"])

    print("\n========== COMPUTATIONAL COST ==========")
    print(f"Baseline NLI evaluations: {baseline_nli_count}")
    print(f"CCEW NLI evaluations: {ccew_nli_count}")
    print(f"Runtime: {runtime:.2f} seconds")

    print("\nSaved:")
    print(summary_path)
    print(claims_path)


if __name__ == "__main__":
    main()
