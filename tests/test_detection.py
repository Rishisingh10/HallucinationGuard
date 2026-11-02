import math

from hallucination_guard.uncertainty import UncertaintyAnalyzer


def test_normalized_entropy_is_between_zero_and_one():
    # Test the mathematical normalization independently of model loading.
    import torch

    logits = torch.tensor([[10.0, 0.0, -2.0]])
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    entropy = -(probs * log_probs).sum(dim=-1)
    normalized = entropy / math.log(logits.shape[-1])

    assert 0.0 <= normalized.item() <= 1.0


def test_single_response_has_zero_semantic_entropy():
    class DummyAnalyzer(UncertaintyAnalyzer):
        def __init__(self):
            pass

    analyzer = DummyAnalyzer()
    result = analyzer.semantic_consistency(["Canberra is the capital of Australia."])

    assert result["semantic_entropy"] == 0.0
    assert result["consistency"] == 1.0
