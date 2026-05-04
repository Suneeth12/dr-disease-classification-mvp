from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import tensorflow as tf

from retina_api.core.settings import CLASS_NAMES
from retina_api.ml import tf_bootstrap  # noqa: F401
from retina_api.ml.patch_mil import build_patch_mil_single_image_inputs, build_retina_mask


DISPLAY_INTERPOLATION = cv2.INTER_LINEAR
GRADCAM_ALPHA = 0.35
GRADCAM_SMOOTHING_SAMPLES = max(1, int(os.getenv("DR_GRADCAM_SMOOTHING_SAMPLES", "1")))
GRADCAM_NOISE_STD = 0.015
FOCUS_CONTOUR_COLOR_RGB = (0.0, 1.0, 0.45)
DEFAULT_EXPLAINABILITY_MODEL_NAMES = ("multiscale",)
PATCH_ATTENTION_NOTE = (
    "Patch-MIL uses the notebook attention weights projected onto retained retinal patches."
)

GRADCAM_TARGET_LAYER_REGISTRY = {
    "multibranch": ["resnet50", "densenet201"],
    "multibranch_model_1": ["resnet50", "densenet201"],
    "attention": ["attention_resnet_apply", "attention_dense_scale"],
    "attention_model": ["attention_resnet_apply", "attention_dense_scale"],
    "multiscale": ["multiscale_resnet_context_fuse", "multiscale_dense_context_fuse"],
    "multiscale_model": ["multiscale_resnet_context_fuse", "multiscale_dense_context_fuse"],
    "lesion": ["lesion_resnet_spatial_apply", "lesion_dense_spatial_apply"],
    "lesion_model": ["lesion_resnet_spatial_apply", "lesion_dense_spatial_apply"],
    "patch_mil": ["patch_mil_dense_td", "patch_mil_resnet_td"],
    "patch_mil_model": ["patch_mil_dense_td", "patch_mil_resnet_td"],
}

EXPLAINABILITY_DISPLAY_NAMES = {
    "multiscale": "Grad-CAM",
    "attention": "Attention",
    "lesion": "Lesion-aware",
    "patch_mil": "Patch-MIL",
}

EXPLAINABILITY_ORDER = [
    "multiscale",
    "attention",
    "lesion",
    "patch_mil",
]


def _tensorflow_device(device_name: str | None):
    return tf.device(device_name) if device_name else nullcontext()


@dataclass(frozen=True)
class ExplainabilityComponent:
    kind: str
    target_layers: tuple[str, ...] = ()
    grad_model: Any | None = None
    attention_probe: Any | None = None


@dataclass
class ExplainabilityComputation:
    artifact: dict[str, Any]
    heatmap: np.ndarray
    processed_rgb: np.ndarray
    retina_mask: np.ndarray


def artifact_sort_key(model_name: str) -> int:
    try:
        return EXPLAINABILITY_ORDER.index(model_name)
    except ValueError:
        return len(EXPLAINABILITY_ORDER)


def display_name_for_model(model_name: str) -> str:
    return EXPLAINABILITY_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())


def resolve_requested_explainability_models(
    ensemble_recipe: dict[str, Any],
    model_names: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    requested_names = list(model_names or DEFAULT_EXPLAINABILITY_MODEL_NAMES)
    available_names = set(str(model_name) for model_name in ensemble_recipe.get("model_names", []))
    selected_names = [model_name for model_name in requested_names if model_name in available_names]
    if not selected_names:
        requested_label = ", ".join(requested_names)
        raise ValueError(f"Requested explainability model is not available: {requested_label}")
    return selected_names


def load_artifact_sidecar(image_path: str | Path) -> dict[str, Any]:
    sidecar_path = Path(image_path).with_suffix(".json")
    if not sidecar_path.exists():
        return {}
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def is_spatial_gradcam_layer(layer: Any) -> bool:
    if isinstance(layer, tf.keras.layers.InputLayer):
        return False
    output_shape = getattr(getattr(layer, "output", None), "shape", None)
    return output_shape is not None and len(output_shape) >= 4


def deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def get_model_call_inputs(model: Any) -> Any:
    if len(model.inputs) == 1:
        return model.input
    return model.inputs


def validate_gradcam_output(heatmap: np.ndarray, expected_min_rank: int = 2) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.size == 0:
        raise ValueError("Grad-CAM heatmap is empty")
    if heatmap.ndim < expected_min_rank:
        raise ValueError(f"Grad-CAM heatmap must have rank >= {expected_min_rank}")
    if not np.all(np.isfinite(heatmap)):
        raise ValueError("Grad-CAM heatmap contains non-finite values")
    return heatmap


def normalize_heatmap_for_display(heatmap: np.ndarray) -> np.ndarray:
    heatmap = validate_gradcam_output(heatmap)
    min_value = float(np.min(heatmap))
    max_value = float(np.max(heatmap))
    if max_value > min_value:
        normalized = (heatmap - min_value) / (max_value - min_value)
    else:
        normalized = np.zeros_like(heatmap, dtype=np.float32)
    return normalized.astype(np.float32)


def describe_heatmap_focus(
    heatmap: np.ndarray,
    retina_mask: np.ndarray,
    *,
    quantile: float = 88,
) -> dict[str, Any]:
    heatmap = normalize_heatmap_for_display(heatmap)
    if heatmap.shape != retina_mask.shape:
        retina_mask = cv2.resize(
            retina_mask.astype(np.float32),
            (heatmap.shape[1], heatmap.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    masked = heatmap * retina_mask.astype(np.float32)
    active_values = masked[retina_mask > 0]
    if active_values.size == 0 or float(np.max(active_values)) <= 0.0:
        return {
            "focus_region": "retinal region",
            "focus_compactness": "unavailable",
            "focus_center": [0.5, 0.5],
            "focus_coverage": 0.0,
            "focus_peak": 0.0,
        }

    threshold = float(np.percentile(active_values, quantile))
    focus_mask = ((masked >= threshold) & (retina_mask > 0)).astype(np.uint8)
    if int(focus_mask.sum()) == 0:
        focus_mask = ((masked == np.max(masked)) & (retina_mask > 0)).astype(np.uint8)

    weights = masked * focus_mask.astype(np.float32)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        center_y, center_x = np.argwhere(focus_mask > 0).mean(axis=0)
    else:
        y_indices, x_indices = np.indices(masked.shape)
        center_y = float(np.sum(y_indices * weights) / weight_sum)
        center_x = float(np.sum(x_indices * weights) / weight_sum)

    height, width = masked.shape
    normalized_x = float(center_x / max(1, width - 1))
    normalized_y = float(center_y / max(1, height - 1))
    horizontal = "left" if normalized_x < 0.34 else "right" if normalized_x > 0.66 else "central"
    vertical = "upper" if normalized_y < 0.34 else "lower" if normalized_y > 0.66 else "mid"
    region = f"{vertical}-{horizontal} retina" if horizontal != "central" else f"{vertical}-central retina"

    retina_area = max(1.0, float(np.sum(retina_mask > 0)))
    coverage = float(np.sum(focus_mask > 0) / retina_area)
    if coverage < 0.08:
        compactness = "localized"
    elif coverage < 0.18:
        compactness = "regional"
    else:
        compactness = "diffuse"

    return {
        "focus_region": region,
        "focus_compactness": compactness,
        "focus_center": [round(normalized_x, 4), round(normalized_y, 4)],
        "focus_coverage": round(coverage, 4),
        "focus_peak": round(float(np.max(masked)), 4),
    }


def refine_heatmap_for_explainability(
    heatmap: np.ndarray,
    retina_mask: np.ndarray,
    quantile: float = 88,
) -> tuple[np.ndarray, np.ndarray]:
    heatmap = normalize_heatmap_for_display(heatmap)
    if heatmap.shape != retina_mask.shape:
        retina_mask = cv2.resize(
            retina_mask.astype(np.float32),
            (heatmap.shape[1], heatmap.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    smoothed = cv2.GaussianBlur(heatmap.astype(np.float32), (9, 9), 0)
    masked = smoothed * retina_mask.astype(np.float32)
    if float(np.max(masked)) > 0.0:
        masked = masked / float(np.max(masked))
    active_values = masked[retina_mask > 0]
    threshold = float(np.percentile(active_values, quantile)) if active_values.size else 1.0
    focus_mask = (masked >= threshold).astype(np.uint8)
    focus_mask = cv2.morphologyEx(focus_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    focus_mask = cv2.dilate(focus_mask, np.ones((3, 3), np.uint8), iterations=1)
    refined = masked * (0.25 + (0.75 * focus_mask.astype(np.float32)))
    refined = refined * retina_mask.astype(np.float32)
    if float(np.max(refined)) > 0.0:
        refined = refined / float(np.max(refined))
    return refined.astype(np.float32), focus_mask.astype(np.uint8)


def add_focus_contours(overlay_rgb: np.ndarray, focus_mask: np.ndarray) -> np.ndarray:
    overlay_uint8 = np.clip(overlay_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    contours, _ = cv2.findContours(focus_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        color = tuple(int(channel * 255) for channel in FOCUS_CONTOUR_COLOR_RGB)
        cv2.drawContours(overlay_uint8, largest_contours, -1, color, thickness=2, lineType=cv2.LINE_AA)
    return overlay_uint8.astype(np.float32) / 255.0


def resolve_gradcam_layer_names(model_name: str, model: Any, override: Any = None) -> list[str]:
    candidates: list[str] = []
    if isinstance(override, (list, tuple)):
        candidates.extend(str(candidate) for candidate in override)
    elif override is not None:
        candidates.append(str(override))

    lookup_keys = [str(model_name).lower(), getattr(model, "name", "").lower()]
    for lookup_key in lookup_keys:
        candidates.extend(GRADCAM_TARGET_LAYER_REGISTRY.get(lookup_key, []))

    ordered_candidates = deduplicate_preserve_order(candidates)
    target_layers: list[str] = []
    for candidate in ordered_candidates:
        try:
            layer = model.get_layer(candidate)
        except ValueError:
            continue
        if is_spatial_gradcam_layer(layer):
            target_layers.append(layer.name)
    if target_layers:
        return deduplicate_preserve_order(target_layers)

    fallback_layers = [
        layer.name
        for layer in reversed(model.layers)
        if is_spatial_gradcam_layer(layer)
    ]
    target_layers = deduplicate_preserve_order(fallback_layers[:2])
    if target_layers:
        return target_layers
    raise ValueError(f"No spatial Grad-CAM layer found for model {model_name!r}")


def resolve_gradcam_output_tensor(model: Any, layer_name: str) -> Any:
    layer = model.get_layer(layer_name)
    if not is_spatial_gradcam_layer(layer):
        raise ValueError(f"Layer {layer_name!r} is not compatible with 2D Grad-CAM")
    if isinstance(layer, tf.keras.Model):
        call_inputs = get_model_call_inputs(model)
        output_tensor = layer(call_inputs)
        if isinstance(output_tensor, (list, tuple)):
            if len(output_tensor) != 1:
                raise ValueError(
                    f"Nested Grad-CAM layer {layer_name!r} returned {len(output_tensor)} tensors"
                )
            output_tensor = output_tensor[0]
        return output_tensor
    return layer.output


def build_gradcam_model(model: Any, layer_names: list[str] | tuple[str, ...]) -> Any:
    target_layers = list(layer_names) if isinstance(layer_names, (list, tuple)) else [str(layer_names)]
    target_tensors = [resolve_gradcam_output_tensor(model, layer_name) for layer_name in target_layers]
    return tf.keras.Model(inputs=model.inputs, outputs=target_tensors + [model.output])


def resolve_gradcam_components(model_name: str, model: Any, override: Any = None) -> tuple[list[str], Any]:
    target_layers = resolve_gradcam_layer_names(model_name, model, override)
    grad_model = build_gradcam_model(model, target_layers)
    return target_layers, grad_model


def normalize_branch_weights(raw_weights: Any, branch_count: int) -> list[float]:
    weights = np.asarray(raw_weights, dtype=np.float32)
    if weights.size != branch_count:
        raise ValueError("Branch-weight count does not match branch heatmaps")
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(branch_count, dtype=np.float32)
    weights = weights / float(np.sum(weights))
    return weights.astype(np.float32).tolist()


def combine_branch_heatmaps(branch_heatmaps: list[np.ndarray], branch_weights: list[float]) -> np.ndarray:
    validated_heatmaps = [normalize_heatmap_for_display(branch_heatmap) for branch_heatmap in branch_heatmaps]
    reference_shape = validated_heatmaps[0].shape
    combined_heatmap = np.zeros(reference_shape, dtype=np.float32)
    for branch_heatmap, branch_weight in zip(validated_heatmaps, branch_weights):
        if branch_heatmap.shape != reference_shape:
            branch_heatmap = cv2.resize(
                branch_heatmap.astype(np.float32),
                (reference_shape[1], reference_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        combined_heatmap += float(branch_weight) * branch_heatmap.astype(np.float32)
    return normalize_heatmap_for_display(combined_heatmap)


def make_gradcam_heatmap(
    img_tensor: np.ndarray,
    grad_model: Any,
    layer_names: list[str] | tuple[str, ...],
    class_index: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    target_layers = list(layer_names) if isinstance(layer_names, (list, tuple)) else [str(layer_names)]
    image_array = np.asarray(img_tensor, dtype=np.float32)
    if image_array.ndim != 4 or image_array.shape[0] != 1:
        raise ValueError("Grad-CAM expects image tensor shape [1, H, W, C]")

    branch_sums: list[np.ndarray | None] = [None] * len(target_layers)
    branch_strengths = np.zeros(len(target_layers), dtype=np.float32)
    rng = np.random.default_rng(42)

    with _tensorflow_device(device_name):
        baseline_outputs = grad_model(tf.convert_to_tensor(image_array, dtype=tf.float32), training=False)
        baseline_preds = baseline_outputs[-1]
        if class_index is None:
            class_index = int(tf.argmax(baseline_preds[0]))

        for sample_index in range(GRADCAM_SMOOTHING_SAMPLES):
            if sample_index == 0:
                sample_input = image_array
            else:
                noise = rng.normal(0.0, GRADCAM_NOISE_STD, image_array.shape).astype(np.float32)
                sample_input = np.clip(image_array + noise, 0.0, 1.0)

            image_tensor = tf.convert_to_tensor(sample_input, dtype=tf.float32)
            with tf.GradientTape() as tape:
                grad_outputs = grad_model(image_tensor, training=False)
                feature_maps = list(grad_outputs[:-1])
                preds = grad_outputs[-1]
                class_channel = preds[:, class_index]

            gradients = tape.gradient(class_channel, feature_maps)
            for branch_index, (feature_map, gradient) in enumerate(zip(feature_maps, gradients)):
                if gradient is None:
                    branch_heatmap = np.zeros(
                        tuple(int(dimension) for dimension in feature_map.shape[1:3]),
                        dtype=np.float32,
                    )
                    branch_strength = 0.0
                else:
                    pooled_gradient = tf.reduce_mean(gradient, axis=(1, 2))[0]
                    branch_heatmap = tf.reduce_sum(
                        feature_map[0] * pooled_gradient[tf.newaxis, tf.newaxis, :],
                        axis=-1,
                    )
                    branch_heatmap = tf.nn.relu(branch_heatmap).numpy().astype(np.float32)
                    branch_heatmap = normalize_heatmap_for_display(branch_heatmap)
                    branch_strength = float(tf.reduce_mean(tf.abs(gradient)).numpy())

                if branch_sums[branch_index] is None:
                    branch_sums[branch_index] = branch_heatmap
                else:
                    branch_sums[branch_index] = branch_sums[branch_index] + branch_heatmap
                branch_strengths[branch_index] += branch_strength

    branch_heatmaps = [
        validate_gradcam_output(branch_sum / float(GRADCAM_SMOOTHING_SAMPLES), expected_min_rank=2)
        for branch_sum in branch_sums
        if branch_sum is not None
    ]
    branch_weights = normalize_branch_weights(
        branch_strengths / float(GRADCAM_SMOOTHING_SAMPLES),
        len(branch_heatmaps),
    )
    combined_heatmap = combine_branch_heatmaps(branch_heatmaps, branch_weights)
    return {
        "combined_heatmap": combined_heatmap,
        "branch_heatmaps": branch_heatmaps,
        "branch_weights": branch_weights,
        "target_layers": target_layers,
        "explained_class": int(class_index),
    }


def build_patch_mil_attention_probe(model: Any) -> Any:
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.output,
            model.get_layer("patch_mil_attention_weights").output,
        ],
        name=f"{model.name}_attention_probe",
    )


def build_patch_mil_attention_heatmap(
    processed_rgb: np.ndarray,
    patch_boxes: np.ndarray,
    attention_weights: np.ndarray,
    patch_valid_mask: np.ndarray,
) -> np.ndarray:
    heatmap = np.zeros(processed_rgb.shape[:2], dtype=np.float32)
    counts = np.zeros_like(heatmap, dtype=np.float32)
    for patch_box, patch_weight, is_valid in zip(patch_boxes, attention_weights, patch_valid_mask):
        if float(is_valid) <= 0.0:
            continue
        x0, y0, x1, y1 = [int(value) for value in patch_box]
        heatmap[y0:y1, x0:x1] += float(patch_weight)
        counts[y0:y1, x0:x1] += 1.0
    counts[counts == 0.0] = 1.0
    return normalize_heatmap_for_display(heatmap / counts)


def overlay_heatmap_on_image(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    retina_mask: np.ndarray | None = None,
    alpha: float = GRADCAM_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    resized_heatmap = cv2.resize(
        heatmap.astype(np.float32),
        (base_image.shape[1], base_image.shape[0]),
        interpolation=DISPLAY_INTERPOLATION,
    )
    display_heatmap = normalize_heatmap_for_display(resized_heatmap)
    resized_mask = None
    if retina_mask is not None:
        resized_mask = cv2.resize(
            retina_mask.astype(np.float32),
            (base_image.shape[1], base_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        display_heatmap = display_heatmap * resized_mask
    heatmap_uint8 = np.clip(display_heatmap * 255.0, 0.0, 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_MAGMA)
    colored_heatmap = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay_base = base_image.astype(np.float32)
    if resized_mask is None:
        overlay = ((1.0 - alpha) * overlay_base) + (alpha * colored_heatmap)
    else:
        mask = resized_mask[..., np.newaxis]
        overlay = ((1.0 - (alpha * mask)) * overlay_base) + ((alpha * mask) * colored_heatmap)
    return np.clip(overlay, 0.0, 1.0), display_heatmap


def _save_overlay_image(output_path: Path, overlay_rgb: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(overlay_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise ValueError(f"Failed to write Grad-CAM artifact: {output_path}")


def _write_sidecar(output_path: Path, payload: dict[str, Any]) -> None:
    serializable_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"image_path"}
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(serializable_payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _base_artifact_payload(
    *,
    model_name: str,
    kind: str,
    output_path: Path,
    target_layers: list[str],
    branch_weights: list[float],
    predicted_class_index: int,
    confidence: float,
    note: str | None,
    member_models: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "display_name": display_name_for_model(model_name),
        "kind": kind,
        "image_path": str(output_path),
        "target_layers": list(target_layers),
        "branch_weights": [float(weight) for weight in branch_weights],
        "predicted_class_index": int(predicted_class_index),
        "predicted_label": CLASS_NAMES[int(predicted_class_index)],
        "confidence": float(confidence),
        "note": note,
        "member_models": list(member_models or []),
    }


def prepare_explainability_component(model_name: str, model: Any) -> ExplainabilityComponent:
    if str(model_name).lower() == "patch_mil":
        return ExplainabilityComponent(
            kind="patch_attention",
            target_layers=("patch_mil_attention_weights",),
            attention_probe=build_patch_mil_attention_probe(model),
        )
    target_layers, grad_model = resolve_gradcam_components(model_name, model)
    return ExplainabilityComponent(
        kind="gradcam",
        target_layers=tuple(target_layers),
        grad_model=grad_model,
    )


def _compute_standard_gradcam_artifact(
    *,
    prepared: dict[str, Any],
    model_name: str,
    component: ExplainabilityComponent,
    probabilities: np.ndarray,
    output_dir: Path,
    device_name: str | None,
) -> ExplainabilityComputation:
    predicted_class_index = int(np.argmax(probabilities))
    gradcam_payload = make_gradcam_heatmap(
        prepared["img_tensor"],
        component.grad_model,
        list(component.target_layers),
        class_index=predicted_class_index,
        device_name=device_name,
    )
    raw_heatmap = validate_gradcam_output(gradcam_payload["combined_heatmap"], expected_min_rank=2)
    retina_mask = build_retina_mask(prepared["processed_rgb"])
    resized_heatmap = cv2.resize(
        raw_heatmap,
        (prepared["processed_rgb"].shape[1], prepared["processed_rgb"].shape[0]),
        interpolation=DISPLAY_INTERPOLATION,
    )
    refined_heatmap, focus_mask = refine_heatmap_for_explainability(resized_heatmap, retina_mask)
    overlay, _ = overlay_heatmap_on_image(
        prepared["processed_rgb"],
        refined_heatmap,
        retina_mask=retina_mask,
    )
    overlay = add_focus_contours(overlay, focus_mask)
    output_path = output_dir / f"{model_name}.png"
    _save_overlay_image(output_path, overlay)
    artifact = _base_artifact_payload(
        model_name=model_name,
        kind="gradcam",
        output_path=output_path,
        target_layers=list(component.target_layers),
        branch_weights=[float(weight) for weight in gradcam_payload["branch_weights"]],
        predicted_class_index=predicted_class_index,
        confidence=float(probabilities[predicted_class_index]),
        note=None,
    )
    artifact.update(describe_heatmap_focus(refined_heatmap, retina_mask))
    _write_sidecar(output_path, artifact)
    return ExplainabilityComputation(
        artifact=artifact,
        heatmap=refined_heatmap,
        processed_rgb=prepared["processed_rgb"],
        retina_mask=retina_mask,
    )


def _compute_patch_attention_artifact(
    *,
    prepared: dict[str, Any],
    model_name: str,
    component: ExplainabilityComponent,
    output_dir: Path,
    device_name: str | None,
) -> ExplainabilityComputation:
    patch_inputs, patch_bag = build_patch_mil_single_image_inputs(prepared["processed_rgb"])
    with _tensorflow_device(device_name):
        probabilities, attention_weights = component.attention_probe(
            {
                key: tf.convert_to_tensor(value, dtype=tf.float32)
                for key, value in patch_inputs.items()
            },
            training=False,
        )
    probabilities = np.asarray(probabilities[0], dtype=np.float32)
    attention_weights = np.asarray(attention_weights[0], dtype=np.float32)
    predicted_class_index = int(np.argmax(probabilities))
    retina_mask = build_retina_mask(prepared["processed_rgb"])
    attention_map = build_patch_mil_attention_heatmap(
        prepared["processed_rgb"],
        patch_inputs["patch_boxes"][0],
        attention_weights,
        patch_inputs["patch_valid_mask"][0],
    )
    refined_heatmap, focus_mask = refine_heatmap_for_explainability(attention_map, retina_mask)
    overlay, _ = overlay_heatmap_on_image(
        prepared["processed_rgb"],
        refined_heatmap,
        retina_mask=retina_mask,
    )
    overlay = add_focus_contours(overlay, focus_mask)
    output_path = output_dir / f"{model_name}.png"
    _save_overlay_image(output_path, overlay)
    artifact = _base_artifact_payload(
        model_name=model_name,
        kind="patch_attention",
        output_path=output_path,
        target_layers=["patch_mil_attention_weights"],
        branch_weights=[1.0],
        predicted_class_index=predicted_class_index,
        confidence=float(probabilities[predicted_class_index]),
        note=PATCH_ATTENTION_NOTE,
    )
    artifact["patch_count_before_padding"] = int(patch_bag["patch_count_before_padding"])
    artifact["patch_count_after_padding"] = int(patch_bag["patch_count_after_padding"])
    artifact.update(describe_heatmap_focus(refined_heatmap, retina_mask))
    _write_sidecar(output_path, artifact)
    return ExplainabilityComputation(
        artifact=artifact,
        heatmap=refined_heatmap,
        processed_rgb=prepared["processed_rgb"],
        retina_mask=retina_mask,
    )


def generate_gradcam_artifacts(
    *,
    prepared: dict[str, Any],
    model_registry: dict[str, Any],
    explainability_registry: dict[str, ExplainabilityComponent],
    ensemble_recipe: dict[str, Any],
    member_probabilities: dict[str, np.ndarray],
    ensemble_prediction: dict[str, Any],
    output_dir: Path,
    device_name: str | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    del model_registry, ensemble_prediction
    output_dir.mkdir(parents=True, exist_ok=True)
    member_results: dict[str, ExplainabilityComputation] = {}
    selected_model_names = resolve_requested_explainability_models(ensemble_recipe, model_names)

    for model_name in selected_model_names:
        if model_name not in explainability_registry:
            raise ValueError(f"Explainability component is not prepared for model: {model_name}")
        if model_name not in member_probabilities:
            raise ValueError(f"Prediction probabilities are not available for model: {model_name}")
        component = explainability_registry[model_name]
        if component.kind == "patch_attention":
            member_results[model_name] = _compute_patch_attention_artifact(
                prepared=prepared,
                model_name=model_name,
                component=component,
                output_dir=output_dir,
                device_name=device_name,
            )
            continue

        member_results[model_name] = _compute_standard_gradcam_artifact(
            prepared=prepared,
            model_name=model_name,
            component=component,
            probabilities=np.asarray(member_probabilities[model_name], dtype=np.float32),
            output_dir=output_dir,
            device_name=device_name,
        )

    artifacts = [result.artifact for result in member_results.values()]
    return sorted(artifacts, key=lambda artifact: artifact_sort_key(str(artifact["model_name"])))
