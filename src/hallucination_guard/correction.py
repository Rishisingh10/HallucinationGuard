
class CorrectionEngine:
    def __init__(
        self,
        evidence_threshold: float = 0.90,
        contradiction_threshold: float = 0.80,
    ):
        self.evidence_threshold = evidence_threshold
        self.contradiction_threshold = contradiction_threshold

    def _abstain(self, original_claim, reason, candidate_attempts=None):
        return {
            "original_claim": original_claim,
            "final_claim": None,
            "action": "ABSTAIN",
            "reason": reason,
            "evidence_id": None,
            "validation_count": 0,
            "candidate_attempts": candidate_attempts or [],
        }

    def correct(self, verification_result: dict, nli_verifier=None):
        original_claim = verification_result["claim"]
        status = verification_result["status"]

        if status == "SUPPORTED":
            return {
                "original_claim": original_claim,
                "final_claim": original_claim,
                "action": "PRESERVE",
                "reason": "The original claim is supported by retrieved evidence.",
                "evidence_id": None,
                "validation_count": 0,
                "candidate_attempts": [],
            }

        if status != "CONTRADICTED":
            return self._abstain(
                original_claim,
                "The evidence is insufficient or conflicting. "
                "A correction cannot be established safely.",
            )

        if nli_verifier is None:
            return self._abstain(
                original_claim,
                "A separate NLI verifier is required to validate correction candidates.",
            )

        evidence = verification_result.get("evidence", [])
        assessments = verification_result.get("nli_assessments", [])

        source_analysis = verification_result.get("source_analysis", {})
        duplicate_ids = {
            item["id"]
            for item in source_analysis.get("duplicate_details", [])
        }

        documents = {
            document["id"]: document
            for document in evidence
        }

        candidates = []

        for assessment in assessments:
            contradiction_score = assessment["scores"]["CONTRADICTION"]

            if (
                assessment["label"] != "CONTRADICTION"
                or contradiction_score < self.contradiction_threshold
            ):
                continue

            if assessment["id"] in duplicate_ids:
                continue

            document = documents.get(assessment["id"])

            if document is not None:
                candidates.append({
                    "document": document,
                    "assessment": assessment,
                })

        candidates.sort(
            key=lambda item: item["document"].get("score", 0),
            reverse=True,
        )

        candidate_attempts = []

        for candidate_item in candidates:
            candidate = candidate_item["document"]
            original_assessment = candidate_item["assessment"]

            candidate_source = candidate.get("source", "UNKNOWN")
            validation_evidence = []

            for document in evidence:
                if document["id"] == candidate["id"]:
                    continue

                if document["id"] in duplicate_ids:
                    continue

                validation_source = document.get("source", "UNKNOWN")

                # Evidence from the candidate's own source
                # cannot independently validate that candidate.
                if (
                    candidate_source == "UNKNOWN"
                    or validation_source == "UNKNOWN"
                    or validation_source == candidate_source
                ):
                    continue

                validation = nli_verifier.verify(
                    candidate["text"],
                    document["text"],
                )

                validation_evidence.append({
                    "evidence_id": document["id"],
                    "source": validation_source,
                    "label": validation["label"],
                    "scores": validation["scores"],
                })

            strong_support = any(
                item["scores"]["ENTAILMENT"] >= self.evidence_threshold
                for item in validation_evidence
            )

            strong_contradiction = any(
                item["scores"]["CONTRADICTION"] >= self.evidence_threshold
                for item in validation_evidence
            )

            independent_sources = sorted({
                item["source"]
                for item in validation_evidence
            })

            accepted = strong_support and not strong_contradiction

            candidate_attempts.append({
                "candidate_id": candidate["id"],
                "candidate_text": candidate["text"],
                "candidate_source": candidate_source,
                "retrieval_score": candidate.get("score", 0),
                "original_claim_assessment": original_assessment,
                "validation_evidence": validation_evidence,
                "independent_sources": independent_sources,
                "independently_supported": strong_support,
                "strong_contradiction_found": strong_contradiction,
                "result": "ACCEPTED" if accepted else "REJECTED",
            })

            if accepted:
                return {
                    "original_claim": original_claim,
                    "final_claim": candidate["text"],
                    "action": "REPLACE_WITH_EVIDENCE",
                    "reason": (
                        "The correction candidate was supported by "
                        "evidence from a different source."
                    ),
                    "evidence_id": candidate["id"],
                    "validation_count": len(validation_evidence),
                    "candidate_attempts": candidate_attempts,
                }

        return self._abstain(
            original_claim,
            "No correction candidate passed independent-source validation.",
            candidate_attempts,
        )
