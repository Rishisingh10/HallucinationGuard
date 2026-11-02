
import ast
import csv
import json
from collections import Counter
from pathlib import Path

from hallucination_guard.adaptive_verification import (
    AdaptiveVerificationEngine,
)
from hallucination_guard.evidence_sufficiency import (
    EvidenceSufficiencyGate,
)


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data/scifact/processed/corpus.json"
RESULTS_PATH = ROOT / "data/scifact/evaluation/claim_results.csv"
OUTPUT_PATH = (
    ROOT
    / "data/scifact/evaluation/esg_rejection_diagnostics.csv"
)

TOP_K = 3

MIN_CLAIM_COVERAGE = 0.25
MIN_NLI_CONFIDENCE = 0.75


def parse_list(value):
    if isinstance(value, list):
        return value

    if not value:
        return []

    return ast.literal_eval(value)


def is_false(value):
    return str(value).strip().lower() in {
        "false", "0", "no"
    }


def normalize_doc_id(value):
    return str(value).replace("scifact_", "")


def load_corpus():
    with CORPUS_PATH.open(encoding="utf-8") as file:
        documents = json.load(file)

    return {
        normalize_doc_id(document["id"]): document
        for document in documents
    }


def load_target_claims():
    targets = []

    with RESULTS_PATH.open(
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if not is_false(row.get("esg_sufficient", "")):
                continue

            gold_ids = set(
                parse_list(row["gold_document_ids"])
            )

            retrieved_ids = set(
                parse_list(row["full_retrieved_ids"])
            )

            gold_hits = gold_ids.intersection(retrieved_ids)

            if not gold_hits:
                continue

            row["gold_hits"] = gold_hits
            targets.append(row)

    return targets


def main():
    print("Loading existing evaluation results...")

    targets = load_target_claims()

    print(f"Rejected claims with gold-document hits: {len(targets)}")

    if not targets:
        print("No matching claims found. Check the CSV columns.")
        return

    print("Loading corpus...")

    documents = load_corpus()

    print("Initializing NLI verifier...")

    engine = AdaptiveVerificationEngine(
        corpus_path=str(CORPUS_PATH),
        max_passages=TOP_K,
        check_entity_coverage=False,
    )

    gate = EvidenceSufficiencyGate(
        min_claim_coverage=MIN_CLAIM_COVERAGE,
        min_nli_confidence=MIN_NLI_CONFIDENCE,
    )

    diagnostic_rows = []
    passage_counts = Counter()
    claim_counts = Counter()
    label_counts = Counter()

    for index, row in enumerate(targets, start=1):
        claim_id = row["id"]
        claim_text = row["claim"]
        gold_label = row["gold_label"]

        retrieved_ids = parse_list(row["full_retrieved_ids"])
        gold_ids = set(parse_list(row["gold_document_ids"]))

        evidence = []

        for doc_id in retrieved_ids:
            document = documents.get(normalize_doc_id(doc_id))

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

        gate_result = gate.evaluate(
            claim_text,
            evidence,
            assessments,
        )

        claim_counts["analyzed"] += 1
        label_counts[gold_label] += 1

        for passage in gate_result["passages"]:
            coverage_pass = (
                passage["coverage"] >= MIN_CLAIM_COVERAGE
            )

            nli_pass = (
                passage["nli_confidence"] >= MIN_NLI_CONFIDENCE
            )

            if coverage_pass and nli_pass:
                reason = "qualified"
            elif not coverage_pass and nli_pass:
                reason = "coverage_only_failure"
            elif coverage_pass and not nli_pass:
                reason = "nli_only_failure"
            else:
                reason = "both_failed"

            passage_counts[reason] += 1

            diagnostic_rows.append({
                "claim_id": claim_id,
                "gold_label": gold_label,
                "claim": claim_text,
                "document_id": passage["id"],
                "is_gold_document": (
                    normalize_doc_id(passage["id"])
                    in gold_ids
                ),
                "coverage": passage["coverage"],
                "entailment": passage["entailment"],
                "contradiction": passage["contradiction"],
                "nli_confidence": passage["nli_confidence"],
                "coverage_pass": coverage_pass,
                "nli_pass": nli_pass,
                "qualified": passage["qualified"],
                "failure_reason": reason,
            })

        if gate_result["sufficient"]:
            claim_counts["now_passes"] += 1
        else:
            claim_counts["still_rejected"] += 1

        if index % 10 == 0 or index == len(targets):
            print(f"Analyzed {index}/{len(targets)} claims")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=diagnostic_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    print("\n========== PASSAGE DIAGNOSTICS ==========")

    for reason, count in passage_counts.items():
        print(f"{reason}: {count}")

    print("\n========== CLAIM DIAGNOSTICS ==========")

    for name, count in claim_counts.items():
        print(f"{name}: {count}")

    print("\n========== GOLD LABELS ==========")

    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("\nSaved diagnostics:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
