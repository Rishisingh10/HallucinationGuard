
import re

from hallucination_guard.retrieval import EvidenceRetriever
from hallucination_guard.nli_verification import NLIVerifier
from hallucination_guard.evidence_aggregation import EvidenceAggregator
from hallucination_guard.risk_features import RiskFeatureExtractor
from hallucination_guard.verification_budget import VerificationBudget
from hallucination_guard.source_analysis import SourceAnalyzer


class AdaptiveVerificationEngine:
    def __init__(
        self,
        corpus_path: str,
        max_passages: int = 3,
        check_entity_coverage: bool = True,
    ):
        self.retriever = EvidenceRetriever(corpus_path)
        self.nli_verifier = NLIVerifier()
        self.aggregator = EvidenceAggregator()
        self.feature_extractor = RiskFeatureExtractor()
        self.source_analyzer = SourceAnalyzer()
        self.check_entity_coverage = check_entity_coverage

        self.budget = VerificationBudget(
            initial_passages=1,
            max_passages=max_passages,
            batch_size=1,
        )

    def _extract_subject(self, claim: str):
        claim = claim.strip()

        if not claim:
            return ""

        words = claim.split()

        predicates = {
            "is", "are", "was", "were",
            "has", "have", "had",
            "founded", "established",
            "released", "created", "built",
            "developed", "acquired", "owns",
            "operates", "produces",
        }

        subject_words = []

        for word in words:
            normalized = re.sub(r"[^\w]", "", word).lower()

            if normalized in predicates:
                break

            subject_words.append(normalized)

        return " ".join(subject_words).strip()

    def _entity_is_covered(self, claim: str):
        subject = self._extract_subject(claim)

        if not subject:
            return False

        corpus_text = " ".join(
            document["text"].lower()
            for document in self.retriever.documents
        )

        return subject in corpus_text

    def _empty_result(self, claim: str, reason: str):
        source_analysis = self.source_analyzer.analyze([])

        return {
            "claim": claim,
            "status": "INSUFFICIENT",
            "evidence": [],
            "nli_assessments": [],
            "source_analysis": source_analysis,
            "features": self.feature_extractor.extract([]),
            "metrics": {
                "retrieval_rounds": 0,
                "passages_retrieved": 0,
                "unique_passages": 0,
                "duplicate_passages": 0,
                "unique_sources": 0,
                "nli_evaluations": 0,
                "maximum_budget": self.budget.max_passages,
            },
            "stop_reason": reason,
        }

    def verify(self, claim: str):
        if not claim or not claim.strip():
            raise ValueError("Claim cannot be empty.")

        if (
            self.check_entity_coverage
            and not self._entity_is_covered(claim)
        ):
            return self._empty_result(
                claim,
                "The claim's subject entity is not covered by the knowledge base.",
            )

        evidence = self.retriever.retrieve(
            claim,
            top_k=self.budget.initial_budget(),
        )

        assessments = []
        assessed_ids = set()
        retrieval_rounds = 0
        stop_reason = "No evidence retrieved."

        while evidence:
            retrieval_rounds += 1

            source_analysis = self.source_analyzer.analyze(evidence)

            duplicate_ids = {
                item["id"]
                for item in source_analysis["duplicate_details"]
            }

            for document in evidence:
                if document["id"] in assessed_ids:
                    continue

                assessed_ids.add(document["id"])

                if document["id"] in duplicate_ids:
                    continue

                assessment = self.nli_verifier.verify(
                    claim,
                    document["text"],
                )

                assessments.append({
                    "id": document["id"],
                    "source": document["source"],
                    "text": document["text"],
                    **assessment,
                })

            aggregation = self.aggregator.aggregate(assessments)
            features = self.feature_extractor.extract(assessments)

            decision = self.budget.next_batch(
                verification_status=aggregation["status"],
                features=features,
                retrieved_count=len(evidence),
            )

            if not decision["retrieve"]:
                stop_reason = decision["reason"]
                break

            next_documents = self.retriever.retrieve(
                claim,
                top_k=decision["count"],
                offset=len(evidence),
            )

            if not next_documents:
                stop_reason = "No additional evidence available."
                break

            evidence.extend(next_documents)

        if not assessments:
            aggregation = {
                "status": "INSUFFICIENT",
                "supporting_count": 0,
                "contradicting_count": 0,
                "neutral_count": 0,
            }

            features = self.feature_extractor.extract([])

        source_analysis = self.source_analyzer.analyze(evidence)

        return {
            "claim": claim,
            "status": aggregation["status"],
            "evidence": evidence,
            "nli_assessments": assessments,
            "source_analysis": source_analysis,
            "features": features,
            "metrics": {
                "retrieval_rounds": retrieval_rounds,
                "passages_retrieved": len(evidence),
                "unique_passages": source_analysis["unique_passages"],
                "duplicate_passages": source_analysis["duplicate_passages"],
                "unique_sources": source_analysis["unique_source_count"],
                "nli_evaluations": len(assessments),
                "maximum_budget": self.budget.max_passages,
            },
            "stop_reason": stop_reason,
        }
