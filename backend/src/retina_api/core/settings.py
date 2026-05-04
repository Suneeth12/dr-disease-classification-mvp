from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMAGE_SIZE = 224
ENSEMBLE_MEMBERS = ["attention", "multiscale", "lesion", "patch_mil"]


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    runtime_dir: Path
    data_dir: Path
    uploads_dir: Path
    artifacts_dir: Path
    models_dir: Path
    notebook_artifacts_dir: Path
    ensemble_recipe_path: Path
    thresholds_dir: Path
    database_url: str


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


def get_settings() -> Settings:
    backend_dir = Path(__file__).resolve().parents[3]
    repo_dir = backend_dir.parent
    runtime_dir = _path_from_env("RETINA_RUNTIME_DIR", repo_dir / "runtime")
    data_dir = _path_from_env("RETINA_DATA_DIR", runtime_dir / "data")
    models_dir = _path_from_env("RETINA_MODELS_DIR", runtime_dir / "models")
    notebook_artifacts_dir = _path_from_env(
        "RETINA_NOTEBOOK_ARTIFACTS_DIR",
        runtime_dir / "notebook-artifacts",
    )
    database_url = os.getenv("RETINA_DATABASE_URL", f"sqlite:///{data_dir / 'app.db'}")

    return Settings(
        base_dir=backend_dir,
        runtime_dir=runtime_dir,
        data_dir=data_dir,
        uploads_dir=data_dir / "uploads",
        artifacts_dir=data_dir / "artifacts",
        models_dir=models_dir,
        notebook_artifacts_dir=notebook_artifacts_dir,
        ensemble_recipe_path=notebook_artifacts_dir / "ensemble" / "final_ensemble_recipe.json",
        thresholds_dir=notebook_artifacts_dir / "thresholds",
        database_url=database_url,
    )


settings = get_settings()
