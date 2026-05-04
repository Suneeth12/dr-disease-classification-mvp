from __future__ import annotations

from typing import Any

import numpy as np


MODEL_DISPLAY_NAMES = {
    "attention": "Attention fusion",
    "patch_mil": "Patch-MIL evidence",
    "lesion": "Lesion-aware fusion",
    "multiscale": "Scale-aware fusion",
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _class_margin(predicted_class_index: int, class_probabilities: list[float]) -> float:
    if not class_probabilities or predicted_class_index >= len(class_probabilities):
        return 0.0
    predicted_probability = float(class_probabilities[predicted_class_index])
    alternatives = [
        float(probability)
        for index, probability in enumerate(class_probabilities)
        if index != predicted_class_index
    ]
    if not alternatives:
        return predicted_probability
    return round(predicted_probability - max(alternatives), 4)


def _normalized_member_weights(
    ensemble_members: list[str],
    ensemble_member_weights: list[float] | None,
) -> list[float]:
    if ensemble_member_weights and len(ensemble_member_weights) == len(ensemble_members):
        weights = np.asarray(ensemble_member_weights, dtype=np.float32)
    else:
        weights = np.ones(len(ensemble_members), dtype=np.float32)
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= 0.0:
        weights = np.ones(len(ensemble_members), dtype=np.float32)
        total = float(np.sum(weights))
    return [round(float(value / total), 4) for value in weights]


def _gradcam_focus(gradcam_artifacts: list[Any]) -> tuple[str, str]:
    artifacts = [_as_dict(artifact) for artifact in gradcam_artifacts]
    gradcam = next(
        (artifact for artifact in artifacts if artifact.get("model_name") == "multiscale"),
        None,
    )
    if not gradcam:
        return "Grad-CAM pending", "generated after visual evidence is requested"
    focus_region = str(gradcam.get("focus_region") or "retinal region")
    focus_pattern = str(gradcam.get("focus_compactness") or "visible")
    return focus_region, focus_pattern


def build_ereg_graph(
    *,
    predicted_class_index: int,
    predicted_label: str,
    confidence: float,
    expected_grade: float,
    class_probabilities: list[float],
    ensemble_members: list[str],
    ensemble_member_weights: list[float] | None,
    threshold_vector: list[float],
    gradcam_artifacts: list[Any],
) -> dict[str, Any]:
    del threshold_vector
    weights = _normalized_member_weights(ensemble_members, ensemble_member_weights)
    focus_region, focus_pattern = _gradcam_focus(gradcam_artifacts)
    class_margin = _class_margin(predicted_class_index, class_probabilities)
    severity_detail = f"Severity score {float(expected_grade):.2f} falls in the {predicted_label} range."

    nodes: list[dict[str, Any]] = [
        {
            "id": "input",
            "label": "Fundus image",
            "kind": "input",
            "detail": "Preprocessed retina image",
        }
    ]
    for model_name, weight in zip(ensemble_members, weights):
        nodes.append(
            {
                "id": model_name,
                "label": MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title()),
                "kind": "member",
                "detail": "E-REG member model",
                "weight": weight,
            }
        )

    nodes.extend(
        [
            {
                "id": "ereg_fusion",
                "label": "E-REG dynamic fusion",
                "kind": "fusion",
                "detail": "Per-image member weights fuse model logits",
            },
            {
                "id": "ordinal_thresholds",
                "label": "Ordinal thresholds",
                "kind": "threshold",
                "detail": severity_detail,
            },
            {
                "id": "final_grade",
                "label": predicted_label,
                "kind": "decision",
                "detail": f"Class {int(predicted_class_index)} at {float(confidence):.1%} confidence",
            },
            {
                "id": "gradcam",
                "label": "Grad-CAM",
                "kind": "visual",
                "detail": f"{focus_pattern} focus in the {focus_region}",
            },
        ]
    )

    edges = [{"source": "input", "target": model_name, "label": "inference"} for model_name in ensemble_members]
    edges.extend(
        [{"source": model_name, "target": "ereg_fusion", "label": "weighted vote"} for model_name in ensemble_members]
    )
    edges.extend(
        [
            {"source": "ereg_fusion", "target": "ordinal_thresholds", "label": "severity score"},
            {"source": "ordinal_thresholds", "target": "final_grade", "label": "grade decision"},
            {"source": "gradcam", "target": "final_grade", "label": "visual focus"},
        ]
    )

    summary = (
        f"E-REG fused {len(ensemble_members)} member models and selected {predicted_label} "
        f"with {float(confidence):.1%} confidence. The severity score was {float(expected_grade):.2f}, "
        f"the class margin was {class_margin:.1%}, and the Grad-CAM focus was "
        f"{focus_pattern} in the {focus_region}."
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "steps": [
            "The uploaded fundus image is preprocessed once for the selected E-REG members.",
            "Each member model contributes a class probability vector for the same image.",
            "E-REG applies dynamic per-image member weights before fusing logits.",
            severity_detail,
            "Only the Grad-CAM visual evidence is displayed for review.",
        ],
        "summary": summary,
        "metrics": {
            "confidence": round(float(confidence), 6),
            "expected_grade": round(float(expected_grade), 4),
            "class_margin": class_margin,
            "predicted_class_index": int(predicted_class_index),
            "member_count": len(ensemble_members),
        },
    }
