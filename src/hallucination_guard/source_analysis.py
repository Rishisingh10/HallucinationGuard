
import re
from difflib import SequenceMatcher
from typing import Dict, List


class SourceAnalyzer:
    def __init__(self, similarity_threshold: float = 0.90):
        self.similarity_threshold = similarity_threshold

    def _normalize(self, text: str) -> str:
        text = text.lower()
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text)).strip()

    def _is_duplicate(self, first: str, second: str) -> bool:
        first = self._normalize(first)
        second = self._normalize(second)

        if first == second:
            return True

        return (
            SequenceMatcher(None, first, second).ratio()
            >= self.similarity_threshold
        )

    def analyze(self, evidence: List[Dict]) -> Dict:
        unique_sources = set()
        unique_passages = []
        duplicate_passages = []

        for document in evidence:
            source = document.get("source", "UNKNOWN")
            text = document.get("text", "")

            unique_sources.add(source)

            duplicate_of = None

            for existing in unique_passages:
                if self._is_duplicate(text, existing["text"]):
                    duplicate_of = existing["id"]
                    break

            if duplicate_of is None:
                unique_passages.append(document)
            else:
                duplicate_passages.append({
                    "id": document.get("id"),
                    "duplicate_of": duplicate_of,
                    "source": source,
                })

        return {
            "retrieved_passages": len(evidence),
            "unique_passages": len(unique_passages),
            "duplicate_passages": len(duplicate_passages),
            "unique_source_count": len(unique_sources),
            "sources": sorted(unique_sources),
            "duplicate_details": duplicate_passages,
        }
