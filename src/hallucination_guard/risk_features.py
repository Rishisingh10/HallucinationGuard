
import math
from typing import Dict, List, Optional


class RiskFeatureExtractor:
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def _normalized_entropy(self, scores: Dict[str, float]) -> float:
        probabilities = [
            scores["ENTAILMENT"],
            scores["CONTRADICTION"],
            scores["NEUTRAL"],
        ]

        entropy = -sum(
            p * math.log(max(p, 1e-12))
            for p in probabilities
        )

        return entropy / math.log(3)

    def extract(
        self,
        assessments: List[Dict],
        generation_uncertainty: Optional[float] = None,
        semantic_disagreement: Optional[float] = None,
    ) -> Dict:
        if not assessments:
            evidence_uncertainty = None
            support_count = 0
            contradiction_count = 0
        else:
            entropies = [
                self._normalized_entropy(item["scores"])
                for item in assessments
            ]

            evidence_uncertainty = sum(entropies) / len(entropies)

            support_count = sum(
                item["scores"]["ENTAILMENT"] >= self.threshold
                for item in assessments
            )

            contradiction_count = sum(
                item["scores"]["CONTRADICTION"] >= self.threshold
                for item in assessments
            )

        conflict = (
            support_count > 0
            and contradiction_count > 0
        )

        return {
            "generation_uncertainty": generation_uncertainty,
            "semantic_disagreement": semantic_disagreement,
            "evidence_uncertainty": evidence_uncertainty,
            "evidence_conflict": conflict,
            "supporting_passages": support_count,
            "contradicting_passages": contradiction_count,
            "retrieved_passages": len(assessments),
        }
