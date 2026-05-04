from pathlib import Path

from retina_api.app import app, create_app
from retina_api.api.dependencies import get_db
from retina_api.core.settings import get_settings
from retina_api.db.models import Base
from retina_api.ml.inference import run_ensemble_inference


def test_backend_package_exports_application_entrypoint() -> None:
    assert app.title == "End-to-End DR Model API"
    assert callable(create_app)
    assert callable(get_db)
    assert Base.metadata.tables
    assert callable(run_ensemble_inference)


def test_default_settings_use_github_friendly_runtime_layout(monkeypatch) -> None:
    monkeypatch.delenv("RETINA_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("RETINA_DATA_DIR", raising=False)
    monkeypatch.delenv("RETINA_MODELS_DIR", raising=False)
    monkeypatch.delenv("RETINA_NOTEBOOK_ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("RETINA_DATABASE_URL", raising=False)

    settings = get_settings()

    assert settings.runtime_dir.name == "runtime"
    assert settings.data_dir == settings.runtime_dir / "data"
    assert settings.models_dir == settings.runtime_dir / "models"
    assert settings.notebook_artifacts_dir == settings.runtime_dir / "notebook-artifacts"
    assert settings.ensemble_recipe_path == settings.notebook_artifacts_dir / "ensemble" / "final_ensemble_recipe.json"


def test_settings_can_be_configured_from_environment(monkeypatch, tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    notebook_dir = tmp_path / "notebook"
    database_url = f"sqlite:///{tmp_path / 'custom.db'}"

    monkeypatch.setenv("RETINA_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("RETINA_DATA_DIR", str(data_dir))
    monkeypatch.setenv("RETINA_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("RETINA_NOTEBOOK_ARTIFACTS_DIR", str(notebook_dir))
    monkeypatch.setenv("RETINA_DATABASE_URL", database_url)

    settings = get_settings()

    assert settings.runtime_dir == runtime_dir
    assert settings.data_dir == data_dir
    assert settings.uploads_dir == data_dir / "uploads"
    assert settings.artifacts_dir == data_dir / "artifacts"
    assert settings.models_dir == models_dir
    assert settings.notebook_artifacts_dir == notebook_dir
    assert settings.ensemble_recipe_path == notebook_dir / "ensemble" / "final_ensemble_recipe.json"
    assert settings.thresholds_dir == notebook_dir / "thresholds"
    assert settings.database_url == database_url
