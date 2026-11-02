
import numpy as np
from sentence_transformers import CrossEncoder


class NLIVerifier:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.model = None

        self.labels = [
            "CONTRADICTION",
            "ENTAILMENT",
            "NEUTRAL",
        ]

    def _load_model(self):
        if self.model is None:
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
            )

    def verify(self, claim: str, evidence: str):
        if not claim.strip() or not evidence.strip():
            raise ValueError("Claim and evidence must not be empty.")

        self._load_model()

        logits = self.model.predict(
            [(evidence, claim)],
            show_progress_bar=False,
        )

        logits = np.asarray(logits).reshape(1, -1)

        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        scores = probabilities[0]
        predicted_index = int(np.argmax(scores))

        return {
            "label": self.labels[predicted_index],
            "scores": {
                label: float(score)
                for label, score in zip(self.labels, scores)
            },
        }
