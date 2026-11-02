
from hallucination_guard.claim_attributes import (
    ClaimAttributeDetector,
)


class ClaimAwareReranker:
    def __init__(self, attribute_bonus: float = 0.01):
        self.detector = ClaimAttributeDetector()
        self.attribute_bonus = attribute_bonus

    def rerank(self, claim, documents):
        claim_attributes = self.detector.detect_claim_attributes(
            claim
        )

        reranked_documents = []

        for document in documents:
            document_attributes = set(
                self.detector.detect_document_attributes(
                    document["text"]
                )
            )

            matched_attributes = [
                attribute
                for attribute in claim_attributes
                if attribute in document_attributes
            ]

            base_score = document["score"]
            bonus = (
                len(matched_attributes) * self.attribute_bonus
            )

            reranked_document = {
                **document,
                "rrf_score": base_score,
                "attribute_matches": matched_attributes,
                "attribute_bonus": bonus,
                "score": base_score + bonus,
            }

            reranked_documents.append(reranked_document)

        return sorted(
            reranked_documents,
            key=lambda document: document["score"],
            reverse=True,
        )
