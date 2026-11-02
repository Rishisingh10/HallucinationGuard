
class VerificationBudget:
    def __init__(
        self,
        initial_passages: int = 1,
        max_passages: int = 3,
        batch_size: int = 1,
        entropy_threshold: float = 0.10,
        min_passages_before_stop: int = 2,
    ):
        if initial_passages < 1:
            raise ValueError("Initial passages must be at least 1.")

        if max_passages < initial_passages:
            raise ValueError("Maximum budget cannot be below initial budget.")

        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")

        if min_passages_before_stop < 1:
            raise ValueError("Minimum passages must be at least 1.")

        if min_passages_before_stop > max_passages:
            raise ValueError(
                "Minimum passages cannot exceed maximum budget."
            )

        self.initial_passages = initial_passages
        self.max_passages = max_passages
        self.batch_size = batch_size
        self.entropy_threshold = entropy_threshold
        self.min_passages_before_stop = min_passages_before_stop

    def initial_budget(self) -> int:
        return min(self.initial_passages, self.max_passages)

    def next_batch(
        self,
        verification_status: str,
        features: dict,
        retrieved_count: int,
    ) -> dict:

        if retrieved_count >= self.max_passages:
            return {
                "retrieve": False,
                "count": 0,
                "reason": "Maximum verification budget reached.",
            }

        if retrieved_count < self.min_passages_before_stop:
            remaining = self.max_passages - retrieved_count

            return {
                "retrieve": True,
                "count": min(self.batch_size, remaining),
                "reason": "Minimum evidence requirement not yet met.",
            }

        evidence_uncertainty = features.get("evidence_uncertainty")
        evidence_conflict = features.get("evidence_conflict", False)

        strong_support = (
            verification_status == "SUPPORTED"
            and evidence_uncertainty is not None
            and evidence_uncertainty <= self.entropy_threshold
            and not evidence_conflict
        )

        if strong_support:
            return {
                "retrieve": False,
                "count": 0,
                "reason": "Strong supporting evidence after minimum evidence requirement.",
            }

        remaining = self.max_passages - retrieved_count

        return {
            "retrieve": True,
            "count": min(self.batch_size, remaining),
            "reason": "Additional evidence is required.",
        }
