import numpy as np

from retina_api.ml.ensemble import normalize_ensemble_weights, weighted_logit_ensemble


def test_normalize_ensemble_weights_returns_uniform_vector_for_zero_sum() -> None:
    weights = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    normalized = normalize_ensemble_weights(weights)

    assert np.allclose(normalized, np.full(4, 0.25, dtype=np.float32))


def test_weighted_logit_ensemble_returns_weighted_softmax_probabilities() -> None:
    logits_a = np.array([[2.0, 0.0, -1.0, -2.0, -3.0]], dtype=np.float32)
    logits_b = np.array([[1.0, 1.5, 0.0, -1.0, -2.0]], dtype=np.float32)

    fused_logits, fused_probs = weighted_logit_ensemble([logits_a, logits_b], weights=[0.75, 0.25])

    assert fused_logits.shape == (1, 5)
    assert fused_probs.shape == (1, 5)
    assert np.allclose(fused_logits, np.array([[1.75, 0.375, -0.75, -1.75, -2.75]], dtype=np.float32))
    assert np.allclose(fused_probs.sum(axis=1), np.array([1.0], dtype=np.float32))
