
from typing import Dict, List


class EvidenceAggregator:
    def __init__(
        self,
        entailment_threshold: float = 0.80,
        contradiction_threshold: float = 0.80,
    ):
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

    def aggregate(self, results: List[Dict]) -> Dict:
        supporting = []
        contradicting = []
        neutral = []

        for result in results:
            scores = result["scores"]

            if (
                result["label"] == "ENTAILMENT"
                and scores["ENTAILMENT"] >= self.entailment_threshold
            ):
                supporting.append(result)

            elif (
                result["label"] == "CONTRADICTION"
                and scores["CONTRADICTION"] >= self.contradiction_threshold
            ):
                contradicting.append(result)

            else:
                neutral.append(result)

        if supporting and contradicting:
            status = "CONFLICTING"
        elif contradicting:
            status = "CONTRADICTED"
        elif supporting:
            status = "SUPPORTED"
        else:
            status = "INSUFFICIENT"

        return {
            "status": status,
            "supporting_count": len(supporting),
            "contradicting_count": len(contradicting),
            "neutral_count": len(neutral),
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
        }
