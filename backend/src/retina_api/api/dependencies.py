from __future__ import annotations

from fastapi import HTTPException, Request

from retina_api.db.session import SessionLocal
from retina_api.ml.model_loader import RuntimeContext


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def runtime_or_503(request: Request) -> RuntimeContext:
    runtime = getattr(request.app.state, "runtime_context", None)
    runtime_status = getattr(request.app.state, "runtime_status", "loading")
    runtime_message = getattr(request.app.state, "runtime_message", None)
    if runtime_status == "error":
        raise HTTPException(status_code=503, detail=runtime_message or "Model runtime failed to load.")
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail=runtime_message or "Backend is still loading models. Please wait and try again.",
        )
    return runtime
