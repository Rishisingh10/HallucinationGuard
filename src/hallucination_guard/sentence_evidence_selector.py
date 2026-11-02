
import re
from typing import Any

from rank_bm25 import BM25Okapi


class SentenceEvidenceSelector:
    """
    Selects claim-relevant sentence windows from retrieved documents.

    This component changes evidence granularity before NLI.
    It does not determine whether a claim is true.
    """

    def __init__(self, window_size: int = 2):
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")

        self.window_size = window_size

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def select(
        self,
        claim: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        if not claim.strip():
            raise ValueError("Claim cannot be empty.")

        if not documents:
            return []

        selected_documents = []
        claim_tokens = self._tokenize(claim)

        for document in documents:
            original_text = document.get("text", "")
            sentences = self._split_sentences(original_text)

            if not sentences:
                selected_documents.append(document.copy())
                continue

            tokenized_sentences = [
                self._tokenize(sentence)
                for sentence in sentences
            ]

            if not any(tokenized_sentences):
                selected_documents.append(document.copy())
                continue

            bm25 = BM25Okapi(tokenized_sentences)
            scores = bm25.get_scores(claim_tokens)

            best_index = int(scores.argmax())

            start = max(
                0,
                best_index - (self.window_size - 1) // 2,
            )

            end = min(
                len(sentences),
                start + self.window_size,
            )

            start = max(
                0,
                end - self.window_size,
            )

            selected_text = " ".join(
                sentences[start:end]
            )

            selected = document.copy()
            selected["text"] = selected_text
            selected["original_text"] = original_text
            selected["selected_sentence_indices"] = list(
                range(start, end)
            )

            selected_documents.append(selected)

        return selected_documents
