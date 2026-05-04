from pathlib import Path

import numpy as np

from retina_api.core.settings import CLASS_NAMES
from retina_api.ml.model_loader import RuntimeContext

from retina_api.ml.inference import predict_single_image_probabilities, run_ensemble_inference


class FakeModel:
    def __init__(self, name: str, probabilities: list[float]) -> None:
        self.name = name
        self._probabilities = np.asarray([probabilities], dtype=np.float32)
        self.last_inputs = None

    def __call__(self, inputs, training: bool = False):
        self.last_inputs = inputs
        return self._probabilities


def test_predict_single_image_probabilities_builds_patch_inputs_for_patch_mil(monkeypatch) -> None:
    model = FakeModel("patch_mil_model", [0.05, 0.10, 0.75, 0.05, 0.05])
    image_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)

    def fake_build_patch_inputs(processed_rgb):
        assert processed_rgb.shape == (224, 224, 3)
        return (
            {
                "patch_images": np.zeros((1, 9, 224, 224, 3), dtype=np.float32),
                "patch_boxes": np.zeros((1, 9, 4), dtype=np.float32),
                "patch_valid_mask": np.ones((1, 9), dtype=np.float32),
            },
            {"patch_count_before_padding": 1},
        )

    monkeypatch.setattr("retina_api.ml.inference.build_patch_mil_single_image_inputs", fake_build_patch_inputs)

    probabilities = predict_single_image_probabilities(model, image_tensor)

    assert probabilities.tolist() == [0.05, 0.10, 0.75, 0.05, 0.05]
    assert isinstance(model.last_inputs, dict)
    assert sorted(model.last_inputs.keys()) == ["patch_boxes", "patch_images", "patch_valid_mask"]


def test_run_ensemble_inference_returns_fast_prediction_payload_by_default(monkeypatch) -> None:
    member_probabilities = [0.05, 0.10, 0.75, 0.05, 0.05]
    runtime = RuntimeContext(
        model_registry={
            "attention": FakeModel("attention_model", member_probabilities),
            "multiscale": FakeModel("multiscale_model", member_probabilities),
            "lesion": FakeModel("lesion_model", member_probabilities),
            "patch_mil": FakeModel("patch_mil_model", member_probabilities),
        },
        explainability_registry={},
        ensemble_recipe={
            "model_names": ["attention", "multiscale", "lesion", "patch_mil"],
            "fusion_method": "weighted_logit_fusion",
            "weights": [0.25, 0.25, 0.25, 0.25],
            "threshold_vector": [0.35, 1.25, 2.25, 3.05],
        },
        thresholds_by_model={},
        artifact_status={},
        artifacts_dir=Path("artifacts"),
    )

    monkeypatch.setattr(
        "retina_api.ml.inference.preprocess_fundus_for_inference",
        lambda image_path: {
            "image_path": str(image_path),
            "original_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "processed_rgb": np.zeros((224, 224, 3), dtype=np.float32),
            "img_tensor": np.zeros((1, 224, 224, 3), dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        "retina_api.ml.inference.build_patch_mil_single_image_inputs",
        lambda processed_rgb: (
            {
                "patch_images": np.zeros((1, 9, 224, 224, 3), dtype=np.float32),
                "patch_boxes": np.zeros((1, 9, 4), dtype=np.float32),
                "patch_valid_mask": np.ones((1, 9), dtype=np.float32),
            },
            {"patch_count_before_padding": 1},
        ),
    )
    def fail_if_gradcam_runs(**kwargs):
        raise AssertionError("Grad-CAM should not run during the fast prediction path")

    monkeypatch.setattr("retina_api.ml.inference.generate_gradcam_artifacts", fail_if_gradcam_runs)

    result = run_ensemble_inference("case.png", runtime)

    assert result["predicted_class_index"] == 2
    assert result["predicted_label"] == CLASS_NAMES[2]
    assert result["ensemble_members"] == ["attention", "multiscale", "lesion", "patch_mil"]
    assert result["class_probabilities"] == member_probabilities
    assert result["confidence"] == member_probabilities[2]
    assert round(result["expected_grade"], 2) == 1.95
    assert result["gradcam_artifacts"] == []


def test_run_ensemble_inference_can_generate_explainability_on_demand(monkeypatch) -> None:
    member_probabilities = [0.05, 0.10, 0.75, 0.05, 0.05]
    runtime = RuntimeContext(
        model_registry={
            "attention": FakeModel("attention_model", member_probabilities),
            "multiscale": FakeModel("multiscale_model", member_probabilities),
            "lesion": FakeModel("lesion_model", member_probabilities),
            "patch_mil": FakeModel("patch_mil_model", member_probabilities),
        },
        explainability_registry={"multiscale": object()},
        ensemble_recipe={
            "model_names": ["attention", "multiscale", "lesion", "patch_mil"],
            "fusion_method": "weighted_logit_fusion",
            "weights": [0.25, 0.25, 0.25, 0.25],
            "threshold_vector": [0.35, 1.25, 2.25, 3.05],
        },
        thresholds_by_model={},
        artifact_status={},
        artifacts_dir=Path("artifacts"),
    )
    requested_explainability_models: list[tuple[str, ...]] = []
    requested_gradcam_models: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        "retina_api.ml.inference.preprocess_fundus_for_inference",
        lambda image_path: {
            "image_path": str(image_path),
            "original_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "processed_rgb": np.zeros((224, 224, 3), dtype=np.float32),
            "img_tensor": np.zeros((1, 224, 224, 3), dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        "retina_api.ml.inference.build_patch_mil_single_image_inputs",
        lambda processed_rgb: (
            {
                "patch_images": np.zeros((1, 9, 224, 224, 3), dtype=np.float32),
                "patch_boxes": np.zeros((1, 9, 4), dtype=np.float32),
                "patch_valid_mask": np.ones((1, 9), dtype=np.float32),
            },
            {"patch_count_before_padding": 1},
        ),
    )
    def fake_ensure_explainability_registry(runtime_arg, model_names=None):
        requested_explainability_models.append(tuple(model_names or []))
        return runtime_arg.explainability_registry

    def fake_generate_gradcam_artifacts(**kwargs):
        requested_gradcam_models.append(tuple(kwargs["model_names"]))
        return [
            {
                "model_name": "multiscale",
                "display_name": "Multiscale",
                "kind": "gradcam",
                "image_path": "artifacts/multiscale.png",
                "target_layers": ["multiscale_resnet_context_fuse", "multiscale_dense_context_fuse"],
                "branch_weights": [0.5, 0.5],
                "predicted_class_index": 2,
                "predicted_label": CLASS_NAMES[2],
                "confidence": 0.75,
                "note": None,
            },
        ]

    monkeypatch.setattr("retina_api.ml.inference.ensure_explainability_registry", fake_ensure_explainability_registry)
    monkeypatch.setattr("retina_api.ml.inference.generate_gradcam_artifacts", fake_generate_gradcam_artifacts)

    result = run_ensemble_inference("case.png", runtime, include_explainability=True)

    assert requested_explainability_models == [("multiscale",)]
    assert requested_gradcam_models == [("multiscale",)]
    assert [artifact["model_name"] for artifact in result["gradcam_artifacts"]] == ["multiscale"]


def test_run_ensemble_inference_passes_runtime_device_to_predictions_and_gradcam(monkeypatch) -> None:
    runtime = RuntimeContext(
        model_registry={
            "multiscale": FakeModel("multiscale_model", [0.05, 0.10, 0.75, 0.05, 0.05]),
        },
        explainability_registry={"multiscale": object()},
        ensemble_recipe={
            "model_names": ["multiscale"],
            "fusion_method": "weighted_logit_fusion",
            "weights": [1.0],
            "threshold_vector": [0.35, 1.25, 2.25, 3.05],
        },
        thresholds_by_model={},
        artifact_status={},
        artifacts_dir=Path("artifacts"),
        tensorflow_device_name="/GPU:0",
    )
    seen_prediction_devices: list[str | None] = []
    seen_gradcam_devices: list[str | None] = []

    monkeypatch.setattr(
        "retina_api.ml.inference.preprocess_fundus_for_inference",
        lambda image_path: {
            "image_path": str(image_path),
            "original_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "processed_rgb": np.zeros((224, 224, 3), dtype=np.float32),
            "img_tensor": np.zeros((1, 224, 224, 3), dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        "retina_api.ml.inference.predict_single_image_probabilities",
        lambda model, inputs, device_name=None: (
            seen_prediction_devices.append(device_name)
            or np.asarray([0.05, 0.10, 0.75, 0.05, 0.05], dtype=np.float32)
        ),
    )
    monkeypatch.setattr(
        "retina_api.ml.inference.ensure_explainability_registry",
        lambda runtime_arg, model_names=None: runtime_arg.explainability_registry,
    )

    def fake_generate_gradcam_artifacts(**kwargs):
        seen_gradcam_devices.append(kwargs.get("device_name"))
        return []

    monkeypatch.setattr("retina_api.ml.inference.generate_gradcam_artifacts", fake_generate_gradcam_artifacts)

    run_ensemble_inference("case.png", runtime, include_explainability=True)

    assert seen_prediction_devices == ["/GPU:0"]
    assert seen_gradcam_devices == ["/GPU:0"]
