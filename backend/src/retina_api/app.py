from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from retina_api.api.routes import router
from retina_api.core.settings import ENSEMBLE_MEMBERS, Settings, settings as default_settings
from retina_api.db.session import init_db
from retina_api.ml.inference import run_ensemble_inference
from retina_api.ml.model_loader import RuntimeContext, load_runtime_context


RuntimeLoader = Callable[[Settings], RuntimeContext]
InferenceRunner = Callable[[str | Path, RuntimeContext], dict[str, Any]]
ExplainabilityRunner = Callable[[str | Path, RuntimeContext], dict[str, Any]]


def run_explainability_inference(image_path: str | Path, runtime: RuntimeContext) -> dict[str, Any]:
    return run_ensemble_inference(image_path, runtime, include_explainability=True)


def create_app(
    *,
    settings: Settings = default_settings,
    runtime_loader: RuntimeLoader = load_runtime_context,
    inference_runner: InferenceRunner = run_ensemble_inference,
    explainability_runner: ExplainabilityRunner = run_explainability_inference,
    load_runtime_in_background: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if settings is default_settings:
            init_db()

        def load_runtime() -> None:
            try:
                runtime = runtime_loader(settings)
            except Exception as exc:  # pragma: no cover - exercised via API behavior
                app.state.runtime_context = None
                app.state.runtime_status = "error"
                app.state.runtime_message = f"Model runtime failed to load: {exc}"
                return

            app.state.runtime_context = runtime
            app.state.runtime_status = "healthy"
            app.state.runtime_message = "Prediction runtime is ready. Grad-CAM is generated on demand."
            app.state.ensemble_members = list(runtime.ensemble_recipe["model_names"])
            app.state.ensemble_recipe = dict(runtime.ensemble_recipe)
            app.state.artifact_status = runtime.artifact_status
            app.state.compute_device = runtime.compute_device

        if load_runtime_in_background:
            runtime_thread = threading.Thread(target=load_runtime, name="runtime-loader", daemon=True)
            app.state.runtime_loader_thread = runtime_thread
            runtime_thread.start()
        else:
            load_runtime()
        yield

    app = FastAPI(title="End-to-End DR Model API", lifespan=lifespan)
    app.state.settings = settings
    app.state.inference_runner = inference_runner
    app.state.explainability_runner = explainability_runner
    app.state.runtime_context = None
    app.state.runtime_status = "loading"
    app.state.runtime_message = "Backend is loading the prediction runtime."
    app.state.ensemble_members = list(ENSEMBLE_MEMBERS)
    app.state.ensemble_recipe = {"model_names": list(ENSEMBLE_MEMBERS), "threshold_vector": []}
    app.state.artifact_status = {}
    app.state.compute_device = {}
    app.state.runtime_loader_thread = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/uploads", StaticFiles(directory=str(settings.uploads_dir), check_dir=False), name="uploads")
    app.mount("/artifacts", StaticFiles(directory=str(settings.artifacts_dir), check_dir=False), name="artifacts")
    app.include_router(router)
    return app


app = create_app()
