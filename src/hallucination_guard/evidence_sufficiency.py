
from typing import Any


class EvidenceSufficiencyGate:
    """
    Checks whether retrieved evidence has sufficient claim coverage
    and NLI confidence.

    This is a heuristic gate, not a calibrated probability model.
    """

    def __init__(
        self,
        min_claim_coverage: float = 0.25,
        min_nli_confidence: float = 0.75,
    ):
        if not 0 <= min_claim_coverage <= 1:
            raise ValueError("min_claim_coverage must be between 0 and 1.")

        if not 0 <= min_nli_confidence <= 1:
            raise ValueError("min_nli_confidence must be between 0 and 1.")

        self.min_claim_coverage = min_claim_coverage
        self.min_nli_confidence = min_nli_confidence

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were",
            "be", "been", "being", "of", "to", "in", "on",
            "for", "and", "or", "but", "with", "by", "as",
            "at", "from", "that", "this", "it", "its",
            "has", "have", "had", "do", "does", "did",
        }

        tokens = set()

        for word in text.lower().split():
            token = word.strip(".,!?;:()[]{}\"'")

            if token and token not in stopwords:
                tokens.add(token)

        return tokens

    def evaluate(
        self,
        claim: str,
        evidence: list[dict[str, Any]],
        assessments: list[dict[str, Any]],
    ) -> dict[str, Any]:

        claim_tokens = self._tokens(claim)

        if not claim_tokens or not evidence:
            return {
                "sufficient": False,
                "reason": "No usable claim or retrieved evidence.",
                "qualified_passages": 0,
                "passages": [],
            }

        assessment_by_id = {
            item["id"]: item
            for item in assessments
            if "id" in item
        }

        passage_results = []

        for document in evidence:
            document_tokens = self._tokens(
                document.get("text", "")
            )

            coverage = (
                len(claim_tokens & document_tokens)
                / len(claim_tokens)
            )

            assessment = assessment_by_id.get(
                document.get("id"), {}
            )

            scores = assessment.get("scores", {})

            entailment = float(
                scores.get("ENTAILMENT", 0.0)
            )

            contradiction = float(
                scores.get("CONTRADICTION", 0.0)
            )

            nli_confidence = max(
                entailment,
                contradiction,
            )

            qualified = (
                coverage >= self.min_claim_coverage
                and nli_confidence >= self.min_nli_confidence
            )

            passage_results.append({
                "id": document.get("id"),
                "coverage": round(coverage, 4),
                "entailment": round(entailment, 4),
                "contradiction": round(contradiction, 4),
                "nli_confidence": round(nli_confidence, 4),
                "qualified": qualified,
            })

        qualified_count = sum(
            result["qualified"]
            for result in passage_results
        )

        return {
            "sufficient": qualified_count > 0,
            "reason": (
                "At least one passage passed both checks."
                if qualified_count
                else "No passage passed both evidence checks."
            ),
            "qualified_passages": qualified_count,
            "passages": passage_results,
        }
