from __future__ import annotations

from typing import Iterable

import numpy as np

import tensorflow as tf

from retina_api.ml import tf_bootstrap  # noqa: F401

SUPPORTED_FUSION_METHODS = {
    "weighted_logit_fusion",
    "ereg_dynamic_logit_fusion",
    "eregpp_dynamic_logit_fusion",
}


def normalize_ensemble_weights(weights: Iterable[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 1:
        raise ValueError("Ensemble weights must be one-dimensional")
    if np.any(weights < 0):
        raise ValueError("Ensemble weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        return np.full_like(weights, 1.0 / len(weights), dtype=np.float32)
    return weights / total


def weighted_logit_ensemble(logit_list: list[np.ndarray], weights: Iterable[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([np.asarray(logits, dtype=np.float32) for logits in logit_list], axis=0)
    if weights is None:
        weights = np.full(stacked.shape[0], 1.0 / stacked.shape[0], dtype=np.float32)
    normalized_weights = normalize_ensemble_weights(weights)
    fused_logits = np.tensordot(normalized_weights, stacked, axes=(0, 0)).astype(np.float32)
    fused_probs = tf.nn.softmax(fused_logits, axis=1).numpy().astype(np.float32)
    return fused_logits, fused_probs


def prediction_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-7, 1.0)
    return -np.sum(probabilities * np.log(probabilities), axis=1) / np.log(probabilities.shape[1])


def top2_margin(probabilities: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(np.asarray(probabilities, dtype=np.float32), axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def confidence_score(probabilities: np.ndarray) -> np.ndarray:
    score = (1.0 - prediction_entropy(probabilities) + top2_margin(probabilities)) / 2.0
    return np.clip(score, 1e-4, 1.0).astype(np.float32)


def expected_grade_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    return probabilities @ np.arange(probabilities.shape[1], dtype=np.float32)


def _recipe_member_weights(model_names: list[str], recipe: dict) -> np.ndarray:
    if "model_reliability" in recipe:
        weights = [float(recipe["model_reliability"].get(model_name, 0.0)) for model_name in model_names]
    elif "model_reliability_scores" in recipe:
        weights = [float(recipe["model_reliability_scores"].get(model_name, 0.0)) for model_name in model_names]
    elif "member_mean_weights_on_validation" in recipe:
        weights = [float(weight) for weight in recipe["member_mean_weights_on_validation"]]
    elif "weights" in recipe:
        weights = [float(weight) for weight in recipe["weights"]]
    else:
        weights = [1.0 for _ in model_names]

    if len(weights) != len(model_names):
        raise ValueError("E-REG recipe member weights must match model_names length")
    return normalize_ensemble_weights(weights)


def ereg_dynamic_member_weights(
    model_names: list[str],
    probabilities_by_model: list[np.ndarray],
    recipe: dict,
) -> np.ndarray:
    if len(model_names) != len(probabilities_by_model):
        raise ValueError("E-REG model_names and probabilities must have the same length")
    probabilities = [np.asarray(probs, dtype=np.float32) for probs in probabilities_by_model]
    sample_count = probabilities[0].shape[0]
    if any(probs.ndim != 2 for probs in probabilities):
        raise ValueError("E-REG probabilities must be [samples, classes]")
    if any(probs.shape[0] != sample_count for probs in probabilities):
        raise ValueError("E-REG probability arrays must have the same sample count")

    reliability_power = float(recipe.get("reliability_power", 1.0))
    confidence_power = float(recipe.get("confidence_power", 0.0))
    agreement_power = float(recipe.get("agreement_power", 0.0))
    base_weights = np.clip(_recipe_member_weights(model_names, recipe), 1e-7, None)
    if "model_reliability" in recipe or "model_reliability_scores" in recipe:
        base_weights = base_weights ** reliability_power
    expected_by_model = np.stack([expected_grade_from_probabilities(probs) for probs in probabilities], axis=1)
    consensus = np.mean(expected_by_model, axis=1, keepdims=True)

    sample_weights = []
    for model_index, probs in enumerate(probabilities):
        confidence = confidence_score(probs) ** confidence_power
        agreement = (1.0 / (1.0 + np.abs(expected_by_model[:, model_index] - consensus[:, 0]))) ** agreement_power
        sample_weights.append(base_weights[model_index] * confidence * agreement)

    weights = np.stack(sample_weights, axis=1).astype(np.float32)
    weights = weights / np.clip(np.sum(weights, axis=1, keepdims=True), 1e-7, None)
    return weights.astype(np.float32)


def ereg_dynamic_logit_ensemble(
    model_names: list[str],
    logit_list: list[np.ndarray],
    probabilities_by_model: list[np.ndarray],
    recipe: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(model_names) != len(logit_list):
        raise ValueError("E-REG model_names and logits must have the same length")
    logits = [np.asarray(values, dtype=np.float32) for values in logit_list]
    member_weights = ereg_dynamic_member_weights(model_names, probabilities_by_model, recipe)
    fused_logits = np.zeros_like(logits[0], dtype=np.float32)
    for model_index, model_logits in enumerate(logits):
        fused_logits += member_weights[:, model_index : model_index + 1] * model_logits
    fused_probs = tf.nn.softmax(fused_logits, axis=1).numpy().astype(np.float32)
    return fused_logits.astype(np.float32), fused_probs, member_weights


def fuse_logits_from_recipe(
    model_names: list[str],
    logit_list: list[np.ndarray],
    probabilities_by_model: list[np.ndarray],
    recipe: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fusion_method = recipe.get("fusion_method", "weighted_logit_fusion")
    if fusion_method == "weighted_logit_fusion":
        raw_weights = recipe.get("weights")
        weights = (
            normalize_ensemble_weights(raw_weights)
            if raw_weights is not None
            else np.full(len(logit_list), 1.0 / len(logit_list), dtype=np.float32)
        )
        fused_logits, fused_probs = weighted_logit_ensemble(logit_list, weights=weights)
        member_weights = np.tile(weights[np.newaxis, :], (fused_logits.shape[0], 1)).astype(np.float32)
        return fused_logits, fused_probs, member_weights
    if fusion_method in {"ereg_dynamic_logit_fusion", "eregpp_dynamic_logit_fusion"}:
        return ereg_dynamic_logit_ensemble(model_names, logit_list, probabilities_by_model, recipe)
    raise ValueError(f"Unsupported ensemble fusion method: {fusion_method}")
