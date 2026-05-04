from pathlib import Path
import threading

import cv2
import numpy as np
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from retina_api.api.dependencies import get_db
from retina_api.api.serializers import artifact_url_for_path, runtime_path_for_static_file, upload_url_for_path
from retina_api.app import create_app
from retina_api.core.settings import Settings
from retina_api.db.models import Artifact, Base, Case, Prediction
from retina_api.ml.model_loader import RuntimeContext



def build_test_session_factory(database_path: Path):
    del database_path
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal


def build_test_settings(tmp_path: Path) -> Settings:
    uploads_dir = tmp_path / "uploads"
    artifacts_dir = tmp_path / "artifacts"
    models_dir = tmp_path / "models"
    notebook_dir = tmp_path / "notebook"
    for path in (uploads_dir, artifacts_dir, models_dir, notebook_dir):
        path.mkdir(parents=True, exist_ok=True)
    return Settings(
        base_dir=tmp_path,
        runtime_dir=tmp_path,
        data_dir=tmp_path / "data",
        uploads_dir=uploads_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        notebook_artifacts_dir=notebook_dir,
        ensemble_recipe_path=notebook_dir / "ensemble" / "final_ensemble_recipe.json",
        thresholds_dir=notebook_dir / "thresholds",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
    )


def build_fake_runtime() -> RuntimeContext:
    return RuntimeContext(
        model_registry={},
        explainability_registry={},
        ensemble_recipe={"model_names": ["attention", "multiscale", "lesion", "patch_mil"]},
        thresholds_by_model={},
        artifact_status={
            "ensemble_recipe": True,
            "model::attention": True,
            "model::multiscale": True,
            "model::lesion": True,
            "model::patch_mil": True,
        },
        artifacts_dir=Path("artifacts"),
        tensorflow_device_name="/GPU:0",
        compute_device={
            "device_name": "/GPU:0",
            "device_type": "GPU",
            "gpu_available": True,
            "physical_gpus": ["/physical_device:GPU:0"],
            "logical_gpus": ["/device:GPU:0"],
            "memory_growth_enabled": True,
            "mixed_precision_policy": "float32",
            "notes": [],
        },
    )


def build_test_image_bytes() -> bytes:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 90, (40, 120, 220), thickness=-1)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


def test_static_urls_preserve_runtime_relative_path_after_project_rename(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    legacy_upload_path = Path("old-project") / "runtime" / "data" / "uploads" / "case-image.png"
    legacy_artifact_path = (
        Path("old-project")
        / "runtime"
        / "data"
        / "artifacts"
        / "case-id"
        / "multiscale.png"
    )

    assert upload_url_for_path(legacy_upload_path, settings=settings) == "/uploads/case-image.png"
    assert artifact_url_for_path(legacy_artifact_path, settings=settings) == "/artifacts/case-id/multiscale.png"

    current_upload_path = settings.uploads_dir / "case-image.png"
    current_upload_path.write_bytes(b"image")
    assert runtime_path_for_static_file(legacy_upload_path, root_dir=settings.uploads_dir) == current_upload_path


def test_health_endpoint_returns_readiness_payload(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    app = create_app(
        settings=settings,
        runtime_loader=lambda settings_arg: build_fake_runtime(),
        inference_runner=lambda image_path, runtime: None,
        load_runtime_in_background=False,
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["ready"] is True
    assert payload["ensemble_members"] == ["attention", "multiscale", "lesion", "patch_mil"]
    assert payload["artifact_status"]["ensemble_recipe"] is True
    assert payload["compute_device"]["device_name"] == "/GPU:0"
    assert payload["compute_device"]["gpu_available"] is True
    assert payload["message"] == "Prediction runtime is ready. Grad-CAM is generated on demand."


def test_predict_and_get_case_return_frontend_ready_payloads(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    session_factory = build_test_session_factory(tmp_path / "app.db")
    multiscale_artifact_path = settings.artifacts_dir / "multiscale.png"
    multiscale_artifact_path.write_bytes(b"multiscale")
    multiscale_artifact_path.with_suffix(".json").write_text(
        '{"display_name":"Multiscale","kind":"gradcam","target_layers":["multiscale_resnet_context_fuse"],'
        '"branch_weights":[0.6,0.4],"predicted_class_index":2,"predicted_label":"Moderate",'
        '"confidence":0.75,"note":null,"focus_region":"upper-central retina",'
        '"focus_compactness":"localized","focus_center":[0.5,0.25],"focus_coverage":0.06}',
        encoding="utf-8",
    )

    def fake_run_inference(image_path, runtime):
        return {
            "predicted_class_index": 2,
            "predicted_label": "Moderate",
            "expected_grade": 1.95,
            "confidence": 0.75,
            "class_probabilities": [0.05, 0.10, 0.75, 0.05, 0.05],
            "ensemble_members": ["attention", "multiscale", "lesion", "patch_mil"],
            "gradcam_artifacts": [
                {
                    "model_name": "multiscale",
                    "display_name": "Multiscale",
                    "kind": "gradcam",
                    "image_path": str(multiscale_artifact_path),
                    "target_layers": ["multiscale_resnet_context_fuse"],
                    "branch_weights": [0.6, 0.4],
                    "predicted_class_index": 2,
                    "predicted_label": "Moderate",
                    "confidence": 0.75,
                    "note": None,
                    "focus_region": "upper-central retina",
                    "focus_compactness": "localized",
                    "focus_center": [0.5, 0.25],
                    "focus_coverage": 0.06,
                },
            ],
        }

    app = create_app(
        settings=settings,
        runtime_loader=lambda settings_arg: build_fake_runtime(),
        inference_runner=fake_run_inference,
        load_runtime_in_background=False,
    )

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/cases/predict",
            data={
                "patient_id": "PAT-42",
                "age": "58",
                "sex": "F",
                "diabetes_duration_years": "12",
                "eye_side": "Left",
                "notes": "blurred vision",
            },
            files={"file": ("fundus.png", build_test_image_bytes(), "image/png")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["prediction"]["predicted_class_index"] == 2
        assert payload["prediction"]["predicted_label"] == "Moderate"
        assert payload["prediction"]["class_probabilities"] == [0.05, 0.10, 0.75, 0.05, 0.05]
        assert payload["prediction"]["ensemble_members"] == ["attention", "multiscale", "lesion", "patch_mil"]
        assert payload["prediction"]["ai_explanation"]["focus_region"] == "upper-central retina"
        assert "Moderate" in payload["prediction"]["ai_explanation"]["summary"]
        assert [artifact["model_name"] for artifact in payload["prediction"]["gradcam_artifacts"]] == ["multiscale"]
        assert payload["prediction"]["ai_explanation"]["supporting_models"] == []
        assert payload["prediction"]["gradcam_artifacts"][0]["image_url"] == "/artifacts/multiscale.png"
        assert payload["original_image_url"].startswith("/uploads/")

        case_response = client.get(f"/api/v1/cases/{payload['case_id']}")

    assert case_response.status_code == 200
    case_payload = case_response.json()
    assert case_payload["case"]["patient_id"] == "PAT-42"
    assert case_payload["prediction"]["class_probabilities"] == [0.05, 0.10, 0.75, 0.05, 0.05]
    assert case_payload["prediction"]["ai_explanation"]["focus_region"] == "upper-central retina"
    assert [artifact["model_name"] for artifact in case_payload["prediction"]["gradcam_artifacts"]] == ["multiscale"]
    assert case_payload["prediction"]["gradcam_artifacts"][0]["kind"] == "gradcam"
    assert case_payload["original_image_url"].startswith("/uploads/")

    with session_factory() as db:
        prediction = db.query(Prediction).filter(Prediction.case_id == payload["case_id"]).one()
        artifacts = db.query(Artifact).filter(Artifact.prediction_id == prediction.id).all()

    assert sorted(artifact.model_name for artifact in artifacts) == ["multiscale"]


def test_predict_rejects_unreadable_image_before_inference(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    session_factory = build_test_session_factory(tmp_path / "app.db")
    inference_calls = 0

    def fake_run_inference(image_path, runtime):
        nonlocal inference_calls
        inference_calls += 1
        return {
            "predicted_class_index": 0,
            "predicted_label": "No DR",
            "expected_grade": 0.0,
            "confidence": 0.95,
            "class_probabilities": [0.95, 0.02, 0.01, 0.01, 0.01],
            "ensemble_members": ["attention", "multiscale", "lesion", "patch_mil"],
            "gradcam_artifacts": [],
        }

    app = create_app(
        settings=settings,
        runtime_loader=lambda settings_arg: build_fake_runtime(),
        inference_runner=fake_run_inference,
        load_runtime_in_background=False,
    )

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/cases/predict",
            data={
                "patient_id": "PAT-BAD",
                "age": "58",
                "sex": "F",
                "diabetes_duration_years": "12",
                "eye_side": "Left",
                "notes": "invalid upload",
            },
            files={"file": ("fundus.png", b"not a real image", "image/png")},
        )

    assert response.status_code == 400
    assert "readable image" in response.json()["detail"]
    assert inference_calls == 0
    assert list(settings.uploads_dir.iterdir()) == []
    with session_factory() as db:
        assert db.query(Case).count() == 0
        assert db.query(Prediction).count() == 0


def test_predict_returns_fast_payload_and_explainability_endpoint_caches_artifacts(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    session_factory = build_test_session_factory(tmp_path / "app.db")
    multiscale_artifact_path = settings.artifacts_dir / "multiscale.png"
    multiscale_artifact_path.write_bytes(b"multiscale")
    multiscale_artifact_path.with_suffix(".json").write_text(
        '{"display_name":"Multiscale","kind":"gradcam","target_layers":["multiscale_resnet_context_fuse"],'
        '"branch_weights":[1.0],"predicted_class_index":3,"predicted_label":"Severe",'
        '"confidence":0.91,"note":null,"focus_region":"mid-central retina"}',
        encoding="utf-8",
    )
    explainability_calls = 0

    def fake_run_prediction(image_path, runtime):
        return {
            "predicted_class_index": 3,
            "predicted_label": "Severe",
            "expected_grade": 3.1,
            "confidence": 0.91,
            "class_probabilities": [0.01, 0.02, 0.03, 0.91, 0.03],
            "ensemble_members": ["attention", "multiscale", "lesion", "patch_mil"],
            "gradcam_artifacts": [],
        }

    def fake_run_explainability(image_path, runtime):
        nonlocal explainability_calls
        explainability_calls += 1
        return {
            "gradcam_artifacts": [
                {
                    "model_name": "multiscale",
                    "display_name": "Multiscale",
                    "kind": "gradcam",
                    "image_path": str(multiscale_artifact_path),
                    "target_layers": ["multiscale_resnet_context_fuse"],
                    "branch_weights": [1.0],
                    "predicted_class_index": 3,
                    "predicted_label": "Severe",
                    "confidence": 0.91,
                    "note": None,
                    "focus_region": "mid-central retina",
                },
            ],
        }

    app = create_app(
        settings=settings,
        runtime_loader=lambda settings_arg: build_fake_runtime(),
        inference_runner=fake_run_prediction,
        explainability_runner=fake_run_explainability,
        load_runtime_in_background=False,
    )

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/cases/predict",
            data={
                "patient_id": "PAT-FAST",
                "age": "60",
                "sex": "M",
                "diabetes_duration_years": "3",
                "eye_side": "Left",
                "notes": "fast path",
            },
            files={"file": ("fundus.png", build_test_image_bytes(), "image/png")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["prediction"]["gradcam_artifacts"] == []

        explainability_response = client.post(f"/api/v1/cases/{payload['case_id']}/explainability")
        cached_response = client.post(f"/api/v1/cases/{payload['case_id']}/explainability")

    assert explainability_response.status_code == 200
    explainability_payload = explainability_response.json()
    assert [artifact["model_name"] for artifact in explainability_payload["prediction"]["gradcam_artifacts"]] == ["multiscale"]
    assert cached_response.status_code == 200
    assert explainability_calls == 1


def test_health_and_predict_show_loading_state_before_runtime_is_ready(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def blocking_runtime_loader(settings_arg: Settings) -> RuntimeContext:
        del settings_arg
        started.set()
        release.wait(timeout=5)
        return build_fake_runtime()

    app = create_app(
        settings=settings,
        runtime_loader=blocking_runtime_loader,
        inference_runner=lambda image_path, runtime: None,
        load_runtime_in_background=True,
    )

    with TestClient(app) as client:
        assert started.wait(timeout=2)

        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["status"] == "loading"
        assert health_payload["ready"] is False
        assert health_payload["message"] == "Backend is loading the prediction runtime."

        predict_response = client.post(
            "/api/v1/cases/predict",
            data={
                "patient_id": "PAT-LOADING",
                "age": "58",
                "sex": "F",
                "diabetes_duration_years": "12",
                "eye_side": "Left",
                "notes": "waiting",
            },
            files={"file": ("fundus.png", build_test_image_bytes(), "image/png")},
        )
        assert predict_response.status_code == 503
        assert "loading the prediction runtime" in predict_response.json()["detail"]

        release.set()
