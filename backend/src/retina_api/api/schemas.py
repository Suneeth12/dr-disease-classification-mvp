from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ready: bool
    ensemble_members: list[str]
    artifact_status: dict[str, bool]
    compute_device: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class GradcamArtifactEnvelope(BaseModel):
    model_name: str
    display_name: str
    kind: str
    image_url: str
    target_layers: list[str]
    branch_weights: list[float]
    predicted_class_index: int
    predicted_label: str
    confidence: float
    note: str | None = None
    member_models: list[str] = Field(default_factory=list)
    patch_count_before_padding: int | None = None
    patch_count_after_padding: int | None = None
    focus_region: str | None = None
    focus_compactness: str | None = None
    focus_center: list[float] = Field(default_factory=list)
    focus_coverage: float | None = None
    focus_peak: float | None = None


class AiExplanationEnvelope(BaseModel):
    summary: str
    focus_region: str
    focus_pattern: str
    model_agreement: str
    agreement_score: float
    supporting_models: list[str] = Field(default_factory=list)
    confidence_reason: str
    class_margin: float
    limitations: str


class EregGraphNodeEnvelope(BaseModel):
    id: str
    label: str
    kind: str
    detail: str
    weight: float | None = None


class EregGraphEdgeEnvelope(BaseModel):
    source: str
    target: str
    label: str


class EregGraphEnvelope(BaseModel):
    nodes: list[EregGraphNodeEnvelope] = Field(default_factory=list)
    edges: list[EregGraphEdgeEnvelope] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    summary: str
    metrics: dict[str, float] = Field(default_factory=dict)


class PredictionEnvelope(BaseModel):
    predicted_class_index: int
    predicted_label: str
    expected_grade: float
    confidence: float
    class_probabilities: list[float]
    ensemble_members: list[str]
    ensemble_member_weights: list[float] = Field(default_factory=list)
    threshold_vector: list[float] = Field(default_factory=list)
    ereg_graph: EregGraphEnvelope
    gradcam_artifacts: list[GradcamArtifactEnvelope]
    ai_explanation: AiExplanationEnvelope


class CaseListItem(BaseModel):
    id: int
    patient_id: str
    visit_date: datetime
    eye_side: str


class CaseDetailResponse(BaseModel):
    case: dict[str, Any]
    prediction: PredictionEnvelope
    original_image_url: str
