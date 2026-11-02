
import ast
import csv
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

from hallucination_guard.adaptive_verification import (
    AdaptiveVerificationEngine,
)
from hallucination_guard.evidence_sufficiency import (
    EvidenceSufficiencyGate,
)


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data/scifact/processed/corpus.json"
RESULTS_PATH = ROOT / "data/scifact/evaluation/claim_results.csv"

OUTPUT_DIR = ROOT / "data/scifact/evaluation"
SUMMARY_PATH = OUTPUT_DIR / "esg_threshold_summary.csv"
DETAILS_PATH = OUTPUT_DIR / "esg_threshold_claim_results.csv"

TOP_K = 3
MIN_CLAIM_COVERAGE = 0.25

NLI_THRESHOLDS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.75]

LABELS = [
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT",
]


def parse_list(value):
    if isinstance(value, list):
        return value

    if not value:
        return []

    return ast.literal_eval(value)


def normalize_doc_id(value):
    return str(value).replace("scifact_", "")


def normalize_label(label):
    label = str(label).strip().upper()

    mapping = {
        "SUPPORT": "SUPPORTED",
        "SUPPORTED": "SUPPORTED",
        "CONTRADICT": "CONTRADICTED",
        "CONTRADICTED": "CONTRADICTED",
        "NOT ENOUGH INFO": "INSUFFICIENT",
        "NOT_ENOUGH_INFO": "INSUFFICIENT",
        "INSUFFICIENT": "INSUFFICIENT",
        "CONFLICTING": "INSUFFICIENT",
    }

    if label not in mapping:
        raise ValueError(f"Unknown label: {label}")

    return mapping[label]


def load_corpus():
    with CORPUS_PATH.open(encoding="utf-8") as file:
        documents = json.load(file)

    return {
        normalize_doc_id(document["id"]): document
        for document in documents
    }


def load_claims():
    with RESULTS_PATH.open(
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def evaluate_threshold(
    threshold,
    claims,
    cached_results,
):
    y_true = []
    y_pred = []
    detail_rows = []

    gate_rejections = 0

    gate = EvidenceSufficiencyGate(
        min_claim_coverage=MIN_CLAIM_COVERAGE,
        min_nli_confidence=threshold,
    )

    for row, cached in zip(claims, cached_results):
        gold_label = normalize_label(row["gold_label"])
        full_prediction = normalize_label(
            row["full_pipeline_prediction"]
        )

        gate_result = gate.evaluate(
            row["claim"],
            cached["evidence"],
            cached["assessments"],
        )

        if gate_result["sufficient"]:
            prediction = full_prediction
        else:
            prediction = "INSUFFICIENT"
            gate_rejections += 1

        y_true.append(gold_label)
        y_pred.append(prediction)

        detail_rows.append({
            "threshold": threshold,
            "claim_id": row["id"],
            "gold_label": gold_label,
            "full_prediction": full_prediction,
            "esg_prediction": prediction,
            "gate_sufficient": gate_result["sufficient"],
            "qualified_passages": gate_result["qualified_passages"],
        })

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    metrics = {
        "nli_threshold": threshold,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "abstention_rate": round(
            y_pred.count("INSUFFICIENT") / len(y_pred),
            4,
        ),
        "gate_rejection_rate": round(
            gate_rejections / len(y_pred),
            4,
        ),
    }

    for index, label in enumerate(LABELS):
        metrics[f"{label.lower()}_precision"] = round(
            precision[index], 4
        )
        metrics[f"{label.lower()}_recall"] = round(
            recall[index], 4
        )
        metrics[f"{label.lower()}_f1"] = round(
            f1[index], 4
        )
        metrics[f"{label.lower()}_support"] = int(
            support[index]
        )

    return metrics, detail_rows


def main():
    print("Loading SciFact evaluation results...")
    claims = load_claims()

    print(f"Claims loaded: {len(claims)}")

    if not claims:
        print("No claims found.")
        return

    print("Loading corpus...")
    documents = load_corpus()

    print("Initializing NLI verifier...")

    engine = AdaptiveVerificationEngine(
        corpus_path=str(CORPUS_PATH),
        max_passages=TOP_K,
        check_entity_coverage=False,
    )

    # Run NLI once per retrieved passage.
    # Reuse these results for every threshold.

    cached_results = []

    total_passages = 0

    for index, row in enumerate(claims, start=1):
        claim_text = row["claim"]
        retrieved_ids = parse_list(row["full_retrieved_ids"])

        evidence = []

        for doc_id in retrieved_ids[:TOP_K]:
            document = documents.get(
                normalize_doc_id(doc_id)
            )

            if document:
                evidence.append(document)

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

        total_passages += len(evidence)

        cached_results.append({
            "evidence": evidence,
            "assessments": assessments,
        })

        if index % 25 == 0 or index == len(claims):
            print(f"Processed {index}/{len(claims)} claims")

    print(f"\nTotal NLI evaluations: {total_passages}")
    print("\nEvaluating thresholds...")

    summary_rows = []
    all_detail_rows = []

    for threshold in NLI_THRESHOLDS:
        metrics, detail_rows = evaluate_threshold(
            threshold,
            claims,
            cached_results,
        )

        summary_rows.append(metrics)
        all_detail_rows.extend(detail_rows)

        print(
            f"Threshold={threshold:.2f} | "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f} | "
            f"Abstention={metrics['abstention_rate']:.2%} | "
            f"Gate rejections={metrics['gate_rejection_rate']:.2%}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with DETAILS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=all_detail_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(all_detail_rows)

    print("\n========== THRESHOLD COMPARISON ==========")

    for metrics in summary_rows:
        print(
            f"\nNLI threshold: {metrics['nli_threshold']}"
        )
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Macro-F1: {metrics['macro_f1']}")
        print(f"Abstention rate: {metrics['abstention_rate']}")
        print(f"Gate rejection rate: {metrics['gate_rejection_rate']}")

        for label in LABELS:
            prefix = label.lower()
            print(
                f"{label}: "
                f"P={metrics[prefix + '_precision']}, "
                f"R={metrics[prefix + '_recall']}, "
                f"F1={metrics[prefix + '_f1']}"
            )

    print("\nSaved summary:")
    print(SUMMARY_PATH)

    print("\nSaved claim-level results:")
    print(DETAILS_PATH)


if __name__ == "__main__":
    main()
