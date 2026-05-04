from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from retina_api.api.schemas import (
    AiExplanationEnvelope,
    CaseDetailResponse,
    EregGraphEnvelope,
    GradcamArtifactEnvelope,
    PredictionEnvelope,
)
from retina_api.core.settings import Settings
from retina_api.db.models import Artifact, Case, Prediction
from retina_api.ml.ereg_graph import build_ereg_graph
from retina_api.ml.explanation import build_ai_explanation
from retina_api.ml.gradcam import (
    DEFAULT_EXPLAINABILITY_MODEL_NAMES,
    artifact_sort_key,
    display_name_for_model,
    load_artifact_sidecar,
)


def static_url_for_path(path: str | Path, *, root_dir: Path, mount_name: str) -> str:
    path_obj = Path(path)
    try:
        relative_path = path_obj.relative_to(root_dir)
    except ValueError:
        root_name = root_dir.name.lower()
        lowered_parts = [part.lower() for part in path_obj.parts]
        if root_name in lowered_parts:
            marker_index = len(lowered_parts) - 1 - lowered_parts[::-1].index(root_name)
            relative_path = Path(*path_obj.parts[marker_index + 1 :])
        else:
            relative_path = Path(path_obj.name)
    return f"/{mount_name}/{relative_path.as_posix()}"


def runtime_path_for_static_file(path: str | Path, *, root_dir: Path) -> Path:
    path_obj = Path(path)
    if path_obj.exists():
        return path_obj
    static_url = static_url_for_path(path_obj, root_dir=root_dir, mount_name=root_dir.name)
    relative_path = Path(static_url.split("/", 2)[-1])
    candidate_path = root_dir / relative_path
    return candidate_path if candidate_path.exists() else path_obj


def upload_url_for_path(path: str | Path, *, settings: Settings) -> str:
    return static_url_for_path(path, root_dir=settings.uploads_dir, mount_name="uploads")


def artifact_url_for_path(path: str | Path, *, settings: Settings) -> str:
    return static_url_for_path(path, root_dir=settings.artifacts_dir, mount_name="artifacts")


def parse_probabilities(raw_value: str | list[float]) -> list[float]:
    if isinstance(raw_value, list):
        return [float(value) for value in raw_value]
    return [float(value) for value in json.loads(raw_value)]


def parse_json_list(raw_value: str | list[Any] | None) -> list[Any]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return raw_value
    return list(json.loads(raw_value))


def parse_json_dict(raw_value: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    return dict(json.loads(raw_value))


def serialize_gradcam_artifact(
    image_path: str | Path,
    model_name: str,
    *,
    settings: Settings,
) -> GradcamArtifactEnvelope:
    resolved_image_path = runtime_path_for_static_file(image_path, root_dir=settings.artifacts_dir)
    metadata = load_artifact_sidecar(resolved_image_path)

    def optional_int(key: str) -> int | None:
        value = metadata.get(key)
        return None if value is None else int(value)

    def optional_float(key: str) -> float | None:
        value = metadata.get(key)
        return None if value is None else float(value)

    return GradcamArtifactEnvelope(
        model_name=model_name,
        display_name=str(metadata.get("display_name", display_name_for_model(model_name))),
        kind=str(metadata.get("kind", "gradcam")),
        image_url=artifact_url_for_path(resolved_image_path, settings=settings),
        target_layers=[str(value) for value in metadata.get("target_layers", [])],
        branch_weights=[float(value) for value in metadata.get("branch_weights", [])],
        predicted_class_index=int(metadata.get("predicted_class_index", 0)),
        predicted_label=str(metadata.get("predicted_label", "")),
        confidence=float(metadata.get("confidence", 0.0)),
        note=str(metadata["note"]) if metadata.get("note") is not None else None,
        member_models=[str(value) for value in metadata.get("member_models", [])],
        patch_count_before_padding=optional_int("patch_count_before_padding"),
        patch_count_after_padding=optional_int("patch_count_after_padding"),
        focus_region=str(metadata["focus_region"]) if metadata.get("focus_region") is not None else None,
        focus_compactness=str(metadata["focus_compactness"]) if metadata.get("focus_compactness") is not None else None,
        focus_center=[float(value) for value in metadata.get("focus_center", [])],
        focus_coverage=optional_float("focus_coverage"),
        focus_peak=optional_float("focus_peak"),
    )


def build_prediction_envelope(
    *,
    predicted_class_index: int,
    predicted_label: str,
    expected_grade: float,
    confidence: float,
    class_probabilities: list[float],
    ensemble_members: list[str],
    ensemble_member_weights: list[float],
    threshold_vector: list[float],
    ereg_graph: dict[str, Any],
    gradcam_artifacts: list[GradcamArtifactEnvelope],
) -> PredictionEnvelope:
    explanation_payload = build_ai_explanation(
        predicted_class_index=predicted_class_index,
        predicted_label=predicted_label,
        confidence=confidence,
        class_probabilities=class_probabilities,
        gradcam_artifacts=[artifact.model_dump() for artifact in gradcam_artifacts],
    )
    graph_payload = ereg_graph or build_ereg_graph(
        predicted_class_index=predicted_class_index,
        predicted_label=predicted_label,
        confidence=confidence,
        expected_grade=expected_grade,
        class_probabilities=class_probabilities,
        ensemble_members=ensemble_members,
        ensemble_member_weights=ensemble_member_weights,
        threshold_vector=threshold_vector,
        gradcam_artifacts=gradcam_artifacts,
    )
    return PredictionEnvelope(
        predicted_class_index=predicted_class_index,
        predicted_label=predicted_label,
        expected_grade=expected_grade,
        confidence=confidence,
        class_probabilities=class_probabilities,
        ensemble_members=ensemble_members,
        ensemble_member_weights=ensemble_member_weights,
        threshold_vector=threshold_vector,
        ereg_graph=EregGraphEnvelope(**graph_payload),
        gradcam_artifacts=gradcam_artifacts,
        ai_explanation=AiExplanationEnvelope(**explanation_payload),
    )


def build_case_detail_response(
    *,
    case: Case,
    prediction: Prediction,
    ensemble_members: list[str],
    ensemble_recipe: dict[str, Any],
    settings: Settings,
    db: Session,
) -> CaseDetailResponse:
    source_image_path = runtime_path_for_static_file(case.source_image_path, root_dir=settings.uploads_dir)
    artifacts = db.query(Artifact).filter(Artifact.prediction_id == prediction.id).all()
    response_artifacts = [
        serialize_gradcam_artifact(artifact.gradcam_image_path, artifact.model_name, settings=settings)
        for artifact in sorted(artifacts, key=lambda item: artifact_sort_key(item.model_name))
        if artifact.model_name in DEFAULT_EXPLAINABILITY_MODEL_NAMES
    ]
    class_probabilities = parse_probabilities(prediction.class_probabilities)
    saved_ensemble_members = [str(value) for value in parse_json_list(prediction.ensemble_members)] or ensemble_members
    ensemble_member_weights = [float(value) for value in parse_json_list(prediction.ensemble_member_weights)]
    threshold_vector = [
        float(value)
        for value in (parse_json_list(prediction.threshold_vector) or ensemble_recipe.get("threshold_vector", []))
    ]
    saved_graph = parse_json_dict(prediction.ereg_graph)
    ereg_graph = build_ereg_graph(
        predicted_class_index=prediction.predicted_class_index,
        predicted_label=prediction.predicted_label,
        confidence=prediction.confidence,
        expected_grade=prediction.expected_grade,
        class_probabilities=class_probabilities,
        ensemble_members=saved_ensemble_members,
        ensemble_member_weights=ensemble_member_weights,
        threshold_vector=threshold_vector,
        gradcam_artifacts=response_artifacts,
    ) if not saved_graph or response_artifacts else saved_graph

    return CaseDetailResponse(
        case={
            "id": case.id,
            "patient_id": case.patient_id,
            "age": case.age,
            "sex": case.sex,
            "diabetes_duration_years": case.diabetes_duration_years,
            "eye_side": case.eye_side,
            "visit_date": case.visit_date,
            "notes": case.notes,
            "source_image_path": str(source_image_path),
        },
        prediction=build_prediction_envelope(
            predicted_class_index=prediction.predicted_class_index,
            predicted_label=prediction.predicted_label,
            expected_grade=prediction.expected_grade,
            confidence=prediction.confidence,
            class_probabilities=class_probabilities,
            ensemble_members=saved_ensemble_members,
            ensemble_member_weights=ensemble_member_weights,
            threshold_vector=threshold_vector,
            ereg_graph=ereg_graph,
            gradcam_artifacts=response_artifacts,
        ),
        original_image_url=upload_url_for_path(source_image_path, settings=settings),
    )


def artifact_payloads_to_models(
    *,
    prediction_id: int,
    artifacts: list[dict[str, Any]],
    db: Session,
) -> list[Artifact]:
    persisted_artifacts: list[Artifact] = []
    for artifact_payload in artifacts:
        db_artifact = Artifact(
            prediction_id=prediction_id,
            model_name=str(artifact_payload["model_name"]),
            gradcam_image_path=str(artifact_payload["image_path"]),
        )
        db.add(db_artifact)
        persisted_artifacts.append(db_artifact)
    db.commit()
    return persisted_artifacts
