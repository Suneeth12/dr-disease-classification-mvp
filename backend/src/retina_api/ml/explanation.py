from __future__ import annotations

from typing import Any


LIMITATION_TEXT = (
    "Grad-CAM is a visual review aid. It shows model focus and does not replace clinical assessment."
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _artifact_dicts(gradcam_artifacts: list[Any]) -> list[dict[str, Any]]:
    artifacts = [_as_dict(artifact) for artifact in gradcam_artifacts]
    return [artifact for artifact in artifacts if artifact.get("focus_region")]


def _select_focus_artifact(gradcam_artifacts: list[Any]) -> dict[str, Any] | None:
    artifacts = _artifact_dicts(gradcam_artifacts)
    for artifact in artifacts:
        if artifact.get("model_name") == "multiscale":
            return artifact
    return artifacts[0] if artifacts else None


def _class_margin(
    predicted_class_index: int,
    class_probabilities: list[float],
) -> tuple[float, str, float]:
    if not class_probabilities or predicted_class_index >= len(class_probabilities):
        return 0.0, "Unknown", 0.0

    predicted_probability = float(class_probabilities[predicted_class_index])
    alternatives = [
        (index, float(probability))
        for index, probability in enumerate(class_probabilities)
        if index != predicted_class_index
    ]
    if not alternatives:
        return predicted_probability, "Unknown", 0.0

    runner_up_index, runner_up_probability = max(alternatives, key=lambda item: item[1])
    margin = round(predicted_probability - runner_up_probability, 4)
    return margin, str(runner_up_index), runner_up_probability


def build_ai_explanation(
    *,
    predicted_class_index: int,
    predicted_label: str,
    confidence: float,
    class_probabilities: list[float],
    gradcam_artifacts: list[Any],
) -> dict[str, Any]:
    focus_artifact = _select_focus_artifact(gradcam_artifacts)
    focus_region = str(focus_artifact.get("focus_region")) if focus_artifact else "retinal region"
    focus_pattern = str(focus_artifact.get("focus_compactness") or "visible") if focus_artifact else "unavailable"
    class_margin, runner_up_index, runner_up_probability = _class_margin(
        predicted_class_index,
        class_probabilities,
    )

    if class_margin >= 0.35:
        confidence_phrase = "a clear probability gap"
    elif class_margin >= 0.12:
        confidence_phrase = "a moderate probability gap"
    elif class_margin >= 0.0:
        confidence_phrase = "a narrow probability gap"
    else:
        confidence_phrase = "an ordinal threshold decision where the expected grade outweighed the raw class peak"

    summary = (
        f"The prediction is {predicted_label}. Grad-CAM highlights the {focus_region} "
        f"with a {focus_pattern} focus pattern. The selected class has {confidence_phrase} over "
        f"the next most likely class (class {runner_up_index}, {runner_up_probability:.1%}), "
        f"with final confidence {float(confidence):.1%}."
    )

    return {
        "summary": summary,
        "focus_region": focus_region,
        "focus_pattern": focus_pattern,
        "model_agreement": "single Grad-CAM" if focus_artifact else "unavailable",
        "agreement_score": 1.0 if focus_artifact else 0.0,
        "supporting_models": [],
        "confidence_reason": (
            f"{predicted_label} was selected with {float(confidence):.1%} confidence. "
            f"The probability margin over the next class is {class_margin:.1%}."
        ),
        "class_margin": round(class_margin, 4),
        "limitations": LIMITATION_TEXT,
    }
