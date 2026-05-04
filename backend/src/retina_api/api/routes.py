from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from retina_api.api.dependencies import get_db, runtime_or_503
from retina_api.api.schemas import CaseDetailResponse, CaseListItem, HealthResponse
from retina_api.api.serializers import (
    artifact_payloads_to_models,
    build_case_detail_response,
    build_prediction_envelope,
    runtime_path_for_static_file,
    serialize_gradcam_artifact,
    upload_url_for_path,
)
from retina_api.db.models import Artifact, Case, Prediction
from retina_api.ml.gradcam import DEFAULT_EXPLAINABILITY_MODEL_NAMES, artifact_sort_key


router = APIRouter(prefix="/api/v1")


def validate_upload_image_bytes(contents: bytes) -> None:
    if not contents:
        raise ValueError("Uploaded file is empty.")
    encoded = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a readable image. Please upload a valid PNG/JPG/JPEG file.")


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    return HealthResponse(
        status=str(request.app.state.runtime_status),
        ready=request.app.state.runtime_status == "healthy",
        ensemble_members=list(request.app.state.ensemble_members),
        artifact_status=dict(request.app.state.artifact_status),
        compute_device=dict(request.app.state.compute_device),
        message=request.app.state.runtime_message,
    )


@router.post("/cases/predict")
async def predict_case(
    request: Request,
    patient_id: str = Form(...),
    age: int = Form(...),
    sex: str = Form(...),
    diabetes_duration_years: int = Form(...),
    eye_side: str = Form(...),
    notes: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    settings = request.app.state.settings
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing upload filename.")
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG/JPG/JPEG allowed.")

    runtime = runtime_or_503(request)
    case_uuid = str(uuid.uuid4())
    extension = Path(file.filename).suffix.lower()
    save_path = settings.uploads_dir / f"{case_uuid}{extension}"

    try:
        upload_contents = await file.read()
        validate_upload_image_bytes(upload_contents)
        save_path.write_bytes(upload_contents)
        prediction_payload = request.app.state.inference_runner(save_path, runtime)
    except ValueError as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        save_path.unlink(missing_ok=True)
        raise

    db_case = Case(
        patient_id=patient_id,
        age=age,
        sex=sex,
        diabetes_duration_years=diabetes_duration_years,
        eye_side=eye_side,
        notes=notes,
        source_image_path=str(save_path),
    )
    db.add(db_case)
    db.commit()
    db.refresh(db_case)

    db_prediction = Prediction(
        case_id=db_case.id,
        predicted_class_index=int(prediction_payload["predicted_class_index"]),
        predicted_label=str(prediction_payload["predicted_label"]),
        expected_grade=float(prediction_payload["expected_grade"]),
        confidence=float(prediction_payload["confidence"]),
        class_probabilities=json.dumps(prediction_payload["class_probabilities"]),
        ensemble_members=json.dumps(prediction_payload.get("ensemble_members", list(runtime.ensemble_recipe["model_names"]))),
        ensemble_member_weights=json.dumps(prediction_payload.get("ensemble_member_weights", [])),
        threshold_vector=json.dumps(prediction_payload.get("threshold_vector", runtime.ensemble_recipe.get("threshold_vector", []))),
        ereg_graph=json.dumps(prediction_payload.get("ereg_graph", {})),
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    persisted_artifacts = artifact_payloads_to_models(
        prediction_id=db_prediction.id,
        artifacts=list(prediction_payload.get("gradcam_artifacts", [])),
        db=db,
    )

    response_artifacts = [
        serialize_gradcam_artifact(artifact.gradcam_image_path, artifact.model_name, settings=settings)
        for artifact in sorted(persisted_artifacts, key=lambda item: artifact_sort_key(item.model_name))
    ]
    response_prediction = build_prediction_envelope(
        predicted_class_index=db_prediction.predicted_class_index,
        predicted_label=db_prediction.predicted_label,
        expected_grade=db_prediction.expected_grade,
        confidence=db_prediction.confidence,
        class_probabilities=prediction_payload["class_probabilities"],
        ensemble_members=list(prediction_payload["ensemble_members"]),
        ensemble_member_weights=list(prediction_payload.get("ensemble_member_weights", [])),
        threshold_vector=list(prediction_payload.get("threshold_vector", runtime.ensemble_recipe.get("threshold_vector", []))),
        ereg_graph=dict(prediction_payload.get("ereg_graph", {})),
        gradcam_artifacts=response_artifacts,
    )

    return {
        "case_id": db_case.id,
        "original_image_url": upload_url_for_path(save_path, settings=settings),
        "prediction": response_prediction.model_dump(),
    }


@router.post("/cases/{case_id}/explainability", response_model=CaseDetailResponse)
def generate_case_explainability(
    case_id: int,
    request: Request,
    db: Session = Depends(get_db),
) -> CaseDetailResponse:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    prediction = db.query(Prediction).filter(Prediction.case_id == case.id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    existing_artifact_count = (
        db.query(Artifact)
        .filter(Artifact.prediction_id == prediction.id)
        .filter(Artifact.model_name.in_(DEFAULT_EXPLAINABILITY_MODEL_NAMES))
        .count()
    )
    if existing_artifact_count == 0:
        runtime = runtime_or_503(request)
        source_image_path = runtime_path_for_static_file(
            case.source_image_path,
            root_dir=request.app.state.settings.uploads_dir,
        )
        try:
            explainability_payload = request.app.state.explainability_runner(source_image_path, runtime)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        artifact_payloads_to_models(
            prediction_id=prediction.id,
            artifacts=list(explainability_payload.get("gradcam_artifacts", [])),
            db=db,
        )

    return build_case_detail_response(
        case=case,
        prediction=prediction,
        ensemble_members=list(request.app.state.ensemble_members),
        ensemble_recipe=dict(request.app.state.ensemble_recipe),
        settings=request.app.state.settings,
        db=db,
    )


@router.get("/cases", response_model=list[CaseListItem])
def list_cases(db: Session = Depends(get_db)) -> list[CaseListItem]:
    cases = db.query(Case).order_by(Case.visit_date.desc()).all()
    return [
        CaseListItem(
            id=case.id,
            patient_id=case.patient_id,
            visit_date=case.visit_date,
            eye_side=case.eye_side,
        )
        for case in cases
    ]


@router.get("/cases/{case_id}", response_model=CaseDetailResponse)
def get_case(case_id: int, request: Request, db: Session = Depends(get_db)) -> CaseDetailResponse:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    prediction = db.query(Prediction).filter(Prediction.case_id == case.id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return build_case_detail_response(
        case=case,
        prediction=prediction,
        ensemble_members=list(request.app.state.ensemble_members),
        ensemble_recipe=dict(request.app.state.ensemble_recipe),
        settings=request.app.state.settings,
        db=db,
    )
