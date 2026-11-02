import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "scifact" / "data"
OUTPUT_DIR = ROOT / "data" / "scifact" / "processed"

CORPUS_FILE = DATA_DIR / "corpus.jsonl"
CLAIMS_FILE = DATA_DIR / "claims_dev.jsonl"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SciFact corpus...")
    raw_corpus = read_jsonl(CORPUS_FILE)

    corpus = {}
    retrieval_documents = []

    for document in raw_corpus:
        doc_id = str(document["doc_id"])
        title = document.get("title", "")
        abstract = document.get("abstract", [])

        text = " ".join([title] + abstract).strip()

        corpus[doc_id] = {
            "title": title,
            "abstract": abstract,
        }

        retrieval_documents.append({
            "id": f"scifact_{doc_id}",
            "source": f"SciFact:{doc_id}",
            "text": text,
        })

    print(f"Loaded {len(corpus)} documents.")

    print("Loading validation claims...")
    raw_claims = read_jsonl(CLAIMS_FILE)

    prepared_claims = []
    label_counts = Counter()
    missing_documents = set()
    mixed_claims = []

    for item in raw_claims:
        evidence = []
        labels = set()

        for doc_id, annotations in item.get("evidence", {}).items():
            doc_id = str(doc_id)

            if doc_id not in corpus:
                missing_documents.add(doc_id)
                continue

            abstract = corpus[doc_id]["abstract"]

            for annotation in annotations:
                label = annotation["label"]
                labels.add(label)

                sentence_indices = annotation.get("sentences", [])
                sentences = []

                for index in sentence_indices:
                    if 0 <= index < len(abstract):
                        sentences.append({
                            "index": index,
                            "text": abstract[index],
                        })

                evidence.append({
                    "doc_id": doc_id,
                    "label": label,
                    "sentences": sentences,
                })

        if not labels:
            gold_label = "NOT ENOUGH INFO"
        elif len(labels) == 1:
            gold_label = next(iter(labels))
        else:
            gold_label = "MIXED"
            mixed_claims.append(item["id"])

        label_counts[gold_label] += 1

        prepared_claims.append({
            "id": item["id"],
            "claim": item["claim"],
            "label": gold_label,
            "gold_evidence": evidence,
            "cited_doc_ids": item.get("cited_doc_ids", []),
        })

    corpus_output = OUTPUT_DIR / "corpus.json"
    claims_output = OUTPUT_DIR / "claims_dev.jsonl"

    with corpus_output.open("w", encoding="utf-8") as file:
        json.dump(retrieval_documents, file, ensure_ascii=False, indent=2)

    write_jsonl(claims_output, prepared_claims)

    print("\n========== PREPARATION SUMMARY ==========")
    print(f"Corpus documents: {len(retrieval_documents)}")
    print(f"Validation claims: {len(prepared_claims)}")

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")

    print(f"\nMixed-label claims: {len(mixed_claims)}")
    print(f"Missing corpus documents: {len(missing_documents)}")

    print("\nGenerated files:")
    print(corpus_output)
    print(claims_output)

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()