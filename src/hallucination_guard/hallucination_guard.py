
from hallucination_guard.adaptive_verification import (
    AdaptiveVerificationEngine,
)
from hallucination_guard.correction import CorrectionEngine


class HallucinationGuard:
    def __init__(self, corpus_path: str, max_passages: int = 3):
        self.verification_engine = AdaptiveVerificationEngine(
            corpus_path=corpus_path,
            max_passages=max_passages,
        )

        self.correction_engine = CorrectionEngine()

    def analyze(self, claim: str) -> dict:
        verification = self.verification_engine.verify(claim)

        correction = self.correction_engine.correct(
            verification,
            nli_verifier=self.verification_engine.nli_verifier,
        )

        return {
            "claim": claim,
            "verification": verification,
            "correction": correction,
        }
