
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class EvaluationMetrics:
    LABELS = [
        "SUPPORTED",
        "CONTRADICTED",
        "INSUFFICIENT",
        "CONFLICTING",
    ]

    def evaluate(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Actual and predicted labels must have equal lengths."
            )

        if not y_true:
            raise ValueError("Evaluation data cannot be empty.")

        valid_labels = set(self.LABELS)

        if not set(y_true).issubset(valid_labels):
            raise ValueError("Invalid label found in actual labels.")

        if not set(y_pred).issubset(valid_labels):
            raise ValueError("Invalid label found in predicted labels.")

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=self.LABELS,
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=self.LABELS,
        )

        per_class = {}

        for index, label in enumerate(self.LABELS):
            per_class[label] = {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
                "support": int(support[index]),
            }

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
            "labels": self.LABELS,
            "total_samples": len(y_true),
        }
