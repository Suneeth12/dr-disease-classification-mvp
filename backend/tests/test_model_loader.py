import json
from pathlib import Path

from retina_api.core.settings import Settings
from retina_api.ml.model_loader import (
    RuntimeContext,
    _patch_serialized_model_config,
    ensure_explainability_registry,
    load_runtime_context,
)
from retina_api.ml.runtime_device import TensorFlowRuntimeInfo


def test_patch_serialized_model_config_rewrites_known_lambda_layers() -> None:
    serialized_config = {
        "config": {
            "layers": [
                {
                    "name": "patch_mil_normalized_boxes",
                    "class_name": "Lambda",
                    "config": {"function": {"class_name": "__lambda__"}},
                },
                {
                    "name": "patch_mil_attention_masked_logits",
                    "class_name": "Lambda",
                    "config": {"function": {"config": "apply_mask"}},
                },
                {
                    "name": "patch_mil_pooled_embedding",
                    "class_name": "Lambda",
                    "config": {"function": {"config": "weighted_sum"}},
                },
                {
                    "name": "attention_resnet_avg_pool",
                    "class_name": "Lambda",
                    "config": {"function": {"class_name": "__lambda__"}},
                },
                {
                    "name": "untouched_layer",
                    "class_name": "Dense",
                    "config": {"units": 5},
                },
            ]
        }
    }

    patched = _patch_serialized_model_config(serialized_config)
    layers_by_name = {layer["name"]: layer for layer in patched["config"]["layers"]}

    assert layers_by_name["patch_mil_normalized_boxes"]["config"]["function"]["config"] == "normalize_boxes"
    assert layers_by_name["patch_mil_normalized_boxes"]["config"]["output_shape"] == [9, 4]
    assert layers_by_name["patch_mil_attention_masked_logits"]["config"]["function"]["config"] == "apply_mask"
    assert layers_by_name["patch_mil_attention_masked_logits"]["config"]["output_shape"] == [9]
    assert layers_by_name["patch_mil_pooled_embedding"]["config"]["function"]["config"] == "weighted_sum"
    assert layers_by_name["patch_mil_pooled_embedding"]["config"]["output_shape"] == [384]
    assert layers_by_name["attention_resnet_avg_pool"]["config"]["function"]["config"] == "spatial_channel_mean"
    assert layers_by_name["attention_resnet_avg_pool"]["config"]["output_shape"] == [7, 7, 1]
    assert "output_shape" not in layers_by_name["untouched_layer"]["config"]


def test_load_runtime_context_loads_recipe_members_and_thresholds(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    notebook_dir = tmp_path / "notebook"
    thresholds_dir = notebook_dir / "thresholds"
    ensemble_dir = notebook_dir / "ensemble"
    uploads_dir = tmp_path / "uploads"
    artifacts_dir = tmp_path / "artifacts"

    for path in (models_dir, thresholds_dir, ensemble_dir, uploads_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)

    recipe = {
        "schema_version": 2,
        "model_names": ["attention", "patch_mil"],
        "fusion_method": "weighted_logit_fusion",
        "weights": [0.5, 0.5],
        "threshold_vector": [0.35, 1.25, 2.25, 3.05],
    }
    (ensemble_dir / "final_ensemble_recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    (thresholds_dir / "attention_thresholds.json").write_text(
        json.dumps({"threshold_name": "attention", "thresholds": [0.35, 1.0, 2.5, 3.5]}),
        encoding="utf-8",
    )
    (thresholds_dir / "patch_mil_thresholds.json").write_text(
        json.dumps({"threshold_name": "patch_mil", "thresholds": [0.35, 1.1, 2.1, 3.3]}),
        encoding="utf-8",
    )
    (models_dir / "attention.keras").write_text("placeholder", encoding="utf-8")
    (models_dir / "patch_mil.keras").write_text("placeholder", encoding="utf-8")

    settings = Settings(
        base_dir=tmp_path,
        runtime_dir=tmp_path,
        data_dir=tmp_path / "data",
        uploads_dir=uploads_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        notebook_artifacts_dir=notebook_dir,
        ensemble_recipe_path=ensemble_dir / "final_ensemble_recipe.json",
        thresholds_dir=thresholds_dir,
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
    )

    calls: list[tuple[str, bool, bool]] = []

    def fake_loader(path: Path, *, compile: bool, safe_mode: bool):
        calls.append((path.name, compile, safe_mode))
        return path.name

    runtime = load_runtime_context(
        settings,
        load_model_fn=fake_loader,
        prepare_explainability_fn=lambda model_name, model: f"gradcam::{model_name}::{model}",
    )

    assert runtime.model_registry == {
        "attention": "attention.keras",
        "patch_mil": "patch_mil.keras",
    }
    assert runtime.explainability_registry == {
        "attention": "gradcam::attention::attention.keras",
        "patch_mil": "gradcam::patch_mil::patch_mil.keras",
    }
    assert runtime.ensemble_recipe["model_names"] == ["attention", "patch_mil"]
    assert runtime.thresholds_by_model["patch_mil"]["threshold_name"] == "patch_mil"
    assert runtime.artifacts_dir == artifacts_dir
    assert calls == [
        ("attention.keras", False, False),
        ("patch_mil.keras", False, False),
    ]


def test_load_runtime_context_configures_tensorflow_before_loading_models(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    notebook_dir = tmp_path / "notebook"
    thresholds_dir = notebook_dir / "thresholds"
    ensemble_dir = notebook_dir / "ensemble"
    uploads_dir = tmp_path / "uploads"
    artifacts_dir = tmp_path / "artifacts"

    for path in (models_dir, thresholds_dir, ensemble_dir, uploads_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)

    recipe = {
        "schema_version": 2,
        "model_names": ["attention"],
        "fusion_method": "weighted_logit_fusion",
        "weights": [1.0],
        "threshold_vector": [0.35, 1.25, 2.25, 3.05],
    }
    (ensemble_dir / "final_ensemble_recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    (thresholds_dir / "attention_thresholds.json").write_text(
        json.dumps({"threshold_name": "attention", "thresholds": [0.35, 1.0, 2.5, 3.5]}),
        encoding="utf-8",
    )
    (models_dir / "attention.keras").write_text("placeholder", encoding="utf-8")

    settings = Settings(
        base_dir=tmp_path,
        runtime_dir=tmp_path,
        data_dir=tmp_path / "data",
        uploads_dir=uploads_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        notebook_artifacts_dir=notebook_dir,
        ensemble_recipe_path=ensemble_dir / "final_ensemble_recipe.json",
        thresholds_dir=thresholds_dir,
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
    )

    calls: list[str] = []

    def fake_configure_tensorflow():
        calls.append("configure")
        return TensorFlowRuntimeInfo(
            device_name="/GPU:0",
            device_type="GPU",
            gpu_available=True,
            physical_gpus=["/physical_device:GPU:0"],
            logical_gpus=["/device:GPU:0"],
            memory_growth_enabled=True,
            mixed_precision_policy="float32",
            notes=[],
        )

    def fake_loader(path: Path, *, compile: bool, safe_mode: bool):
        del compile, safe_mode
        calls.append(f"load::{path.name}")
        return path.name

    runtime = load_runtime_context(
        settings,
        load_model_fn=fake_loader,
        configure_tensorflow_fn=fake_configure_tensorflow,
    )

    assert calls == ["configure", "load::attention.keras"]
    assert runtime.tensorflow_device_name == "/GPU:0"
    assert runtime.compute_device["device_type"] == "GPU"
    assert runtime.artifact_status == {
        "ensemble_recipe": True,
        "model::attention": True,
        "threshold::attention": True,
    }


def test_ensure_explainability_registry_prepares_only_requested_models() -> None:
    runtime = RuntimeContext(
        model_registry={
            "attention": "attention-model",
            "multiscale": "multiscale-model",
            "lesion": "lesion-model",
            "patch_mil": "patch-model",
        },
        explainability_registry={},
        ensemble_recipe={"model_names": ["attention", "multiscale", "lesion", "patch_mil"]},
        thresholds_by_model={},
        artifact_status={},
        artifacts_dir=Path("artifacts"),
    )
    prepared: list[tuple[str, str]] = []

    def fake_prepare(model_name: str, model: str) -> str:
        prepared.append((model_name, model))
        return f"explain::{model_name}"

    registry = ensure_explainability_registry(
        runtime,
        prepare_explainability_fn=fake_prepare,
        model_names=["multiscale"],
    )

    assert registry == {"multiscale": "explain::multiscale"}
    assert prepared == [("multiscale", "multiscale-model")]
    assert runtime.artifact_status == {"explainability::multiscale": True}
