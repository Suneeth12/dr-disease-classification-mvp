from __future__ import annotations

import json
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from zipfile import ZipFile

import keras
import tensorflow as tf

from retina_api.core.settings import IMAGE_SIZE, Settings
from retina_api.ml import tf_bootstrap  # noqa: F401
from retina_api.ml.ensemble import SUPPORTED_FUSION_METHODS
from retina_api.ml.gradcam import prepare_explainability_component
from retina_api.ml.runtime_device import TensorFlowRuntimeInfo, configure_tensorflow_runtime


LoadModelFn = Callable[..., Any]
PrepareExplainabilityFn = Callable[[str, Any], Any]
ConfigureTensorFlowFn = Callable[[], TensorFlowRuntimeInfo]


LAMBDA_LAYER_PATCHES: dict[str, dict[str, Any]] = {
    "attention_resnet_avg_pool": {
        "function_name": "spatial_channel_mean",
        "output_shape": [7, 7, 1],
    },
    "attention_resnet_max_pool": {
        "function_name": "spatial_channel_max",
        "output_shape": [7, 7, 1],
    },
    "lesion_dense_spatial_avg_pool": {
        "function_name": "spatial_channel_mean",
        "output_shape": [7, 7, 1],
    },
    "lesion_dense_spatial_max_pool": {
        "function_name": "spatial_channel_max",
        "output_shape": [7, 7, 1],
    },
    "lesion_resnet_spatial_avg_pool": {
        "function_name": "spatial_channel_mean",
        "output_shape": [7, 7, 1],
    },
    "lesion_resnet_spatial_max_pool": {
        "function_name": "spatial_channel_max",
        "output_shape": [7, 7, 1],
    },
    "patch_mil_normalized_boxes": {
        "function_name": "normalize_boxes",
        "output_shape": [9, 4],
    },
    "patch_mil_attention_masked_logits": {
        "function_name": "apply_mask",
        "output_shape": [9],
    },
    "patch_mil_pooled_embedding": {
        "function_name": "weighted_sum",
        "output_shape": [384],
    },
    "patch_mil_attention_summary": {
        "function_name": "summarize_attention",
        "output_shape": [1],
    },
}


def spatial_channel_mean(tensor: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tensor, axis=-1, keepdims=True)


def spatial_channel_max(tensor: tf.Tensor) -> tf.Tensor:
    return tf.reduce_max(tensor, axis=-1, keepdims=True)


def normalize_boxes(boxes: tf.Tensor) -> tf.Tensor:
    return boxes / tf.cast(IMAGE_SIZE, boxes.dtype)


def apply_mask(values: list[tf.Tensor] | tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    logits, valid_mask = values
    valid_mask = tf.cast(valid_mask, logits.dtype)
    large_negative = tf.constant(-1e4, dtype=logits.dtype)
    return tf.where(valid_mask > 0.0, logits, large_negative)


def weighted_sum(values: list[tf.Tensor] | tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    embeddings, weights = values
    return tf.reduce_sum(embeddings * weights, axis=1)


def summarize_attention(weights: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(weights, axis=1, keepdims=True)


CUSTOM_OBJECTS: dict[str, Any] = {
    "spatial_channel_mean": spatial_channel_mean,
    "spatial_channel_max": spatial_channel_max,
    "normalize_boxes": normalize_boxes,
    "apply_mask": apply_mask,
    "weighted_sum": weighted_sum,
    "summarize_attention": summarize_attention,
}


@dataclass(frozen=True)
class RuntimeContext:
    model_registry: dict[str, Any]
    explainability_registry: dict[str, Any]
    ensemble_recipe: dict[str, Any]
    thresholds_by_model: dict[str, dict[str, Any]]
    artifact_status: dict[str, bool]
    artifacts_dir: Path
    tensorflow_device_name: str = "/CPU:0"
    compute_device: dict[str, Any] = field(default_factory=dict)
    explainability_lock: Any = field(default_factory=threading.Lock)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_keras_archive_config(path: Path) -> dict[str, Any]:
    with ZipFile(path) as archive:
        return json.loads(archive.read("config.json"))


def _keras_function_config(function_name: str) -> dict[str, str]:
    return {
        "module": "builtins",
        "class_name": "function",
        "config": function_name,
        "registered_name": "function",
    }


def _patch_serialized_model_config(serialized_config: dict[str, Any]) -> dict[str, Any]:
    patched_config = deepcopy(serialized_config)
    layers = patched_config.get("config", {}).get("layers", [])

    for layer in layers:
        if layer.get("class_name") != "Lambda":
            continue

        layer_patch = LAMBDA_LAYER_PATCHES.get(layer.get("name"))
        if not layer_patch:
            continue

        layer_config = layer.setdefault("config", {})
        layer_config["function"] = _keras_function_config(layer_patch["function_name"])
        layer_config["output_shape"] = list(layer_patch["output_shape"])

    return patched_config


def _load_model_from_patched_archive(path: Path) -> Any:
    serialized_config = _patch_serialized_model_config(_read_keras_archive_config(path))
    model = keras.saving.deserialize_keras_object(
        serialized_config,
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False,
    )
    model.load_weights(path)
    return model


def _default_load_model(path: Path, *, compile: bool, safe_mode: bool) -> Any:
    del compile, safe_mode
    return _load_model_from_patched_archive(path)


def load_runtime_context(
    settings: Settings,
    load_model_fn: LoadModelFn | None = None,
    prepare_explainability_fn: PrepareExplainabilityFn | None = None,
    configure_tensorflow_fn: ConfigureTensorFlowFn | None = None,
    eager_explainability: bool | None = None,
) -> RuntimeContext:
    load_model = load_model_fn or _default_load_model
    prepare_explainability = prepare_explainability_fn or prepare_explainability_component
    configure_tensorflow = configure_tensorflow_fn or (lambda: configure_tensorflow_runtime(tf_module=tf))
    should_prepare_explainability = eager_explainability if eager_explainability is not None else prepare_explainability_fn is not None

    recipe_path = settings.ensemble_recipe_path
    if not recipe_path.exists():
        raise FileNotFoundError(f"Missing ensemble recipe: {recipe_path}")

    ensemble_recipe = _read_json(recipe_path)
    model_names = ensemble_recipe.get("model_names", [])
    if not model_names:
        raise ValueError("Ensemble recipe does not define any model names")
    fusion_method = ensemble_recipe.get("fusion_method", "weighted_logit_fusion")
    if fusion_method not in SUPPORTED_FUSION_METHODS:
        supported = ", ".join(sorted(SUPPORTED_FUSION_METHODS))
        raise ValueError(f"Unsupported ensemble fusion method {fusion_method!r}. Supported: {supported}")
    if fusion_method in {"ereg_dynamic_logit_fusion", "eregpp_dynamic_logit_fusion"}:
        mean_weights = ensemble_recipe.get("member_mean_weights_on_validation", [])
        if mean_weights and len(mean_weights) != len(model_names):
            raise ValueError("E-REG recipe member_mean_weights_on_validation must match model_names")
    if len(ensemble_recipe.get("threshold_vector", [])) != 4:
        raise ValueError("Ensemble recipe threshold_vector must define 4 thresholds")

    tensorflow_runtime = configure_tensorflow()
    model_registry: dict[str, Any] = {}
    explainability_registry: dict[str, Any] = {}
    thresholds_by_model: dict[str, dict[str, Any]] = {}
    artifact_status: dict[str, bool] = {"ensemble_recipe": True}

    for model_name in model_names:
        model_path = settings.models_dir / f"{model_name}.keras"
        threshold_path = settings.thresholds_dir / f"{model_name}_thresholds.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        if not threshold_path.exists():
            raise FileNotFoundError(f"Missing threshold artifact: {threshold_path}")

        with tf.device(tensorflow_runtime.device_name):
            model_registry[model_name] = load_model(model_path, compile=False, safe_mode=False)
            if should_prepare_explainability:
                explainability_registry[model_name] = prepare_explainability(model_name, model_registry[model_name])
        thresholds_by_model[model_name] = _read_json(threshold_path)
        artifact_status[f"model::{model_name}"] = True
        artifact_status[f"threshold::{model_name}"] = True
        if should_prepare_explainability:
            artifact_status[f"explainability::{model_name}"] = True

    return RuntimeContext(
        model_registry=model_registry,
        explainability_registry=explainability_registry,
        ensemble_recipe=ensemble_recipe,
        thresholds_by_model=thresholds_by_model,
        artifact_status=artifact_status,
        artifacts_dir=settings.artifacts_dir,
        tensorflow_device_name=tensorflow_runtime.device_name,
        compute_device=tensorflow_runtime.as_dict(),
        explainability_lock=threading.Lock(),
    )


def ensure_explainability_registry(
    runtime: RuntimeContext,
    prepare_explainability_fn: PrepareExplainabilityFn | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    prepare_explainability = prepare_explainability_fn or prepare_explainability_component
    requested_model_names = list(model_names or runtime.ensemble_recipe["model_names"])
    with runtime.explainability_lock:
        for model_name in requested_model_names:
            if model_name in runtime.explainability_registry:
                continue
            if model_name not in runtime.model_registry:
                raise ValueError(f"Explainability model is not loaded: {model_name}")
            with tf.device(runtime.tensorflow_device_name):
                runtime.explainability_registry[model_name] = prepare_explainability(
                    model_name,
                    runtime.model_registry[model_name],
                )
            runtime.artifact_status[f"explainability::{model_name}"] = True
    return runtime.explainability_registry
