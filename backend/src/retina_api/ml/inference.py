from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

import tensorflow as tf

from retina_api.core.settings import CLASS_NAMES
from retina_api.ml import tf_bootstrap  # noqa: F401
from retina_api.ml.ensemble import fuse_logits_from_recipe
from retina_api.ml.ereg_graph import build_ereg_graph
from retina_api.ml.gradcam import DEFAULT_EXPLAINABILITY_MODEL_NAMES, generate_gradcam_artifacts
from retina_api.ml.model_loader import RuntimeContext, ensure_explainability_registry
from retina_api.ml.patch_mil import build_patch_mil_single_image_inputs
from retina_api.ml.preprocessing import preprocess_fundus_for_inference
from retina_api.ml.thresholds import apply_thresholds, logits_from_probabilities


def _tensorflow_device(device_name: str | None):
    return tf.device(device_name) if device_name else nullcontext()


def predict_single_image_probabilities(
    model: Any,
    model_inputs: Any,
    *,
    device_name: str | None = None,
) -> np.ndarray:
    with _tensorflow_device(device_name):
        if isinstance(model_inputs, dict):
            prepared_inputs = {
                key: tf.convert_to_tensor(value, dtype=tf.float32)
                for key, value in model_inputs.items()
            }
            outputs = model(prepared_inputs, training=False)
        else:
            if getattr(model, "name", "").lower() == "patch_mil_model":
                image_array = np.asarray(model_inputs, dtype=np.float32)
                if image_array.ndim == 4:
                    image_array = image_array[0]
                patch_inputs, _ = build_patch_mil_single_image_inputs(image_array)
                prepared_inputs = {
                    key: tf.convert_to_tensor(value, dtype=tf.float32)
                    for key, value in patch_inputs.items()
                }
                outputs = model(prepared_inputs, training=False)
            else:
                tensor = tf.convert_to_tensor(model_inputs, dtype=tf.float32)
                outputs = model(tensor, training=False)
    return np.round(np.asarray(outputs[0], dtype=np.float64), 6)


def run_ensemble_inference(
    image_path: str | Path,
    runtime: RuntimeContext,
    *,
    include_explainability: bool = False,
) -> dict[str, Any]:
    image_path = Path(image_path)
    prepared = preprocess_fundus_for_inference(image_path)
    member_probs: list[np.ndarray] = []
    member_logits: list[np.ndarray] = []
    member_probabilities: dict[str, np.ndarray] = {}

    for model_name in runtime.ensemble_recipe["model_names"]:
        probabilities = predict_single_image_probabilities(
            runtime.model_registry[model_name],
            prepared["img_tensor"],
            device_name=runtime.tensorflow_device_name,
        )
        probs_2d = probabilities[np.newaxis, :]
        member_probs.append(probs_2d)
        member_logits.append(logits_from_probabilities(probs_2d))
        member_probabilities[model_name] = probabilities

    _, fused_probs, member_weights = fuse_logits_from_recipe(
        list(runtime.ensemble_recipe["model_names"]),
        member_logits,
        member_probs,
        runtime.ensemble_recipe,
    )
    ensemble_probs = np.round(fused_probs[0].astype(np.float64), 6)
    expected_grade = float(np.dot(ensemble_probs, np.arange(len(ensemble_probs), dtype=np.float64)))

    threshold_vector = np.asarray(runtime.ensemble_recipe["threshold_vector"], dtype=np.float32)
    predicted_class_index = int(
        apply_thresholds(np.asarray([expected_grade], dtype=np.float32), threshold_vector)[0]
    )
    confidence = float(ensemble_probs[predicted_class_index])
    gradcam_artifacts: list[dict[str, Any]] = []

    if include_explainability:
        artifact_output_dir = runtime.artifacts_dir / image_path.stem
        explainability_registry = ensure_explainability_registry(
            runtime,
            model_names=DEFAULT_EXPLAINABILITY_MODEL_NAMES,
        )
        gradcam_artifacts = generate_gradcam_artifacts(
            prepared=prepared,
            model_registry=runtime.model_registry,
            explainability_registry=explainability_registry,
            ensemble_recipe=runtime.ensemble_recipe,
            member_probabilities=member_probabilities,
            ensemble_prediction={
                "predicted_class_index": predicted_class_index,
                "confidence": confidence,
            },
            output_dir=artifact_output_dir,
            device_name=runtime.tensorflow_device_name,
            model_names=DEFAULT_EXPLAINABILITY_MODEL_NAMES,
        )

    ensemble_member_weights = [float(value) for value in member_weights[0].tolist()]
    threshold_values = [round(float(value), 6) for value in threshold_vector.tolist()]
    ereg_graph = build_ereg_graph(
        predicted_class_index=predicted_class_index,
        predicted_label=CLASS_NAMES[predicted_class_index],
        confidence=confidence,
        expected_grade=expected_grade,
        class_probabilities=[float(value) for value in ensemble_probs.tolist()],
        ensemble_members=list(runtime.ensemble_recipe["model_names"]),
        ensemble_member_weights=ensemble_member_weights,
        threshold_vector=threshold_values,
        gradcam_artifacts=gradcam_artifacts,
    )

    return {
        "predicted_class_index": predicted_class_index,
        "predicted_label": CLASS_NAMES[predicted_class_index],
        "expected_grade": expected_grade,
        "confidence": confidence,
        "class_probabilities": [float(value) for value in ensemble_probs.tolist()],
        "ensemble_members": list(runtime.ensemble_recipe["model_names"]),
        "ensemble_member_weights": ensemble_member_weights,
        "threshold_vector": threshold_values,
        "ereg_graph": ereg_graph,
        "gradcam_artifacts": gradcam_artifacts,
    }
