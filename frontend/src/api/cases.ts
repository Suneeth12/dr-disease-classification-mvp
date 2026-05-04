import { fetchJson } from './client';
import type {
  CaseDetailResponse,
  CaseListItem,
  EregGraph,
  EregGraphEdge,
  EregGraphNode,
  GradcamArtifact,
  HealthResponse,
  ModelExplanation,
  PredictResponse,
} from './types';

export const CLASS_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'];
export const EXPLAINABILITY_MODEL_NAME = 'multiscale';

export const EMPTY_MODEL_EXPLANATION: ModelExplanation = {
  summary: 'The model explanation is unavailable for this case.',
  focus_region: 'Unavailable',
  focus_pattern: 'Unavailable',
  confidence_reason: 'Confidence reasoning is unavailable for this case.',
  class_margin: 0,
};

export const EMPTY_EREG_GRAPH: EregGraph = {
  nodes: [],
  edges: [],
  steps: [],
  summary: 'E-REG graph explanation is unavailable for this case.',
  metrics: {},
};

export function getHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>('/api/v1/health');
}

export function createPrediction(formData: FormData): Promise<PredictResponse> {
  return fetchJson<PredictResponse>('/api/v1/cases/predict', {
    method: 'POST',
    body: formData,
  });
}

export function listCases(signal: AbortSignal): Promise<CaseListItem[]> {
  return fetchJson<CaseListItem[]>('/api/v1/cases', { signal });
}

export async function getCaseDetail(caseId: number, signal: AbortSignal): Promise<CaseDetailResponse> {
  const payload = await fetchJson<unknown>(`/api/v1/cases/${caseId}`, { signal });
  return normalizeCaseDetailResponse(payload);
}

export async function generateCaseExplainability(caseId: number): Promise<CaseDetailResponse> {
  const payload = await fetchJson<unknown>(`/api/v1/cases/${caseId}/explainability`, {
    method: 'POST',
  });
  return normalizeCaseDetailResponse(payload);
}

function normalizeGradcamArtifact(input: unknown): GradcamArtifact | null {
  if (!input || typeof input !== 'object') {
    return null;
  }

  const record = input as Record<string, unknown>;
  return {
    model_name: typeof record.model_name === 'string' ? record.model_name : 'unknown',
    display_name: typeof record.display_name === 'string' ? record.display_name : 'Grad-CAM',
    image_url: typeof record.image_url === 'string' ? record.image_url : '',
  };
}

function normalizeModelExplanation(input: unknown): ModelExplanation {
  if (!input || typeof input !== 'object') {
    return EMPTY_MODEL_EXPLANATION;
  }

  const record = input as Record<string, unknown>;
  return {
    summary: typeof record.summary === 'string' ? record.summary : EMPTY_MODEL_EXPLANATION.summary,
    focus_region: typeof record.focus_region === 'string' ? record.focus_region : EMPTY_MODEL_EXPLANATION.focus_region,
    focus_pattern: typeof record.focus_pattern === 'string' ? record.focus_pattern : EMPTY_MODEL_EXPLANATION.focus_pattern,
    confidence_reason: typeof record.confidence_reason === 'string'
      ? record.confidence_reason
      : EMPTY_MODEL_EXPLANATION.confidence_reason,
    class_margin: Number.isFinite(Number(record.class_margin)) ? Number(record.class_margin) : 0,
  };
}

function normalizeStringArray(input: unknown): string[] {
  return Array.isArray(input) ? input.map((value) => String(value)) : [];
}

function normalizeNumberArray(input: unknown): number[] {
  return Array.isArray(input)
    ? input.map((value) => Number(value)).filter((value) => Number.isFinite(value))
    : [];
}

function normalizeEregGraphNode(input: unknown): EregGraphNode | null {
  if (!input || typeof input !== 'object') {
    return null;
  }
  const record = input as Record<string, unknown>;
  const id = typeof record.id === 'string' ? record.id : '';
  const label = typeof record.label === 'string' ? record.label : '';
  if (!id || !label) {
    return null;
  }
  const weight = Number(record.weight);
  return {
    id,
    label,
    kind: typeof record.kind === 'string' ? record.kind : 'unknown',
    detail: typeof record.detail === 'string' ? record.detail : '',
    weight: Number.isFinite(weight) ? weight : null,
  };
}

function normalizeEregGraphEdge(input: unknown): EregGraphEdge | null {
  if (!input || typeof input !== 'object') {
    return null;
  }
  const record = input as Record<string, unknown>;
  const source = typeof record.source === 'string' ? record.source : '';
  const target = typeof record.target === 'string' ? record.target : '';
  if (!source || !target) {
    return null;
  }
  return {
    source,
    target,
    label: typeof record.label === 'string' ? record.label : '',
  };
}

function normalizeEregGraph(input: unknown): EregGraph {
  if (!input || typeof input !== 'object') {
    return EMPTY_EREG_GRAPH;
  }
  const record = input as Record<string, unknown>;
  const rawMetrics = record.metrics && typeof record.metrics === 'object'
    ? record.metrics as Record<string, unknown>
    : {};
  const metrics = Object.fromEntries(
    Object.entries(rawMetrics)
      .map(([key, value]) => [key, Number(value)] as const)
      .filter(([, value]) => Number.isFinite(value)),
  );

  return {
    nodes: Array.isArray(record.nodes)
      ? record.nodes.map((node) => normalizeEregGraphNode(node)).filter((node): node is EregGraphNode => node !== null)
      : [],
    edges: Array.isArray(record.edges)
      ? record.edges.map((edge) => normalizeEregGraphEdge(edge)).filter((edge): edge is EregGraphEdge => edge !== null)
      : [],
    steps: Array.isArray(record.steps) ? record.steps.map((step) => String(step)) : [],
    summary: typeof record.summary === 'string' ? record.summary : EMPTY_EREG_GRAPH.summary,
    metrics,
  };
}

function normalizeCaseDetailResponse(payload: unknown): CaseDetailResponse {
  const record = (payload && typeof payload === 'object' ? payload : {}) as Record<string, unknown>;
  const predictionRecord = (record.prediction && typeof record.prediction === 'object'
    ? record.prediction
    : {}) as Record<string, unknown>;

  return {
    case: (record.case && typeof record.case === 'object' ? record.case : {}) as CaseDetailResponse['case'],
    original_image_url: typeof record.original_image_url === 'string' ? record.original_image_url : '',
    prediction: {
      predicted_class_index: Number.isFinite(Number(predictionRecord.predicted_class_index))
        ? Number(predictionRecord.predicted_class_index)
        : 0,
      predicted_label: typeof predictionRecord.predicted_label === 'string' ? predictionRecord.predicted_label : 'Unknown',
      expected_grade: Number.isFinite(Number(predictionRecord.expected_grade)) ? Number(predictionRecord.expected_grade) : 0,
      confidence: Number.isFinite(Number(predictionRecord.confidence)) ? Number(predictionRecord.confidence) : 0,
      class_probabilities: Array.isArray(predictionRecord.class_probabilities)
        ? predictionRecord.class_probabilities.map((value) => Number(value)).filter((value) => Number.isFinite(value))
        : [],
      ensemble_members: normalizeStringArray(predictionRecord.ensemble_members),
      ensemble_member_weights: normalizeNumberArray(predictionRecord.ensemble_member_weights),
      threshold_vector: normalizeNumberArray(predictionRecord.threshold_vector),
      ereg_graph: normalizeEregGraph(predictionRecord.ereg_graph),
      gradcam_artifacts: Array.isArray(predictionRecord.gradcam_artifacts)
        ? predictionRecord.gradcam_artifacts
            .map((artifact) => normalizeGradcamArtifact(artifact))
            .filter((artifact): artifact is GradcamArtifact => artifact !== null)
        : [],
      ai_explanation: normalizeModelExplanation(predictionRecord.ai_explanation),
    },
  };
}
