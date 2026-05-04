export interface HealthResponse {
  status: string;
  ready: boolean;
  artifact_status: Record<string, boolean>;
  compute_device?: {
    device_name?: string;
    device_type?: string;
    gpu_available?: boolean;
  };
  message?: string | null;
}

export interface PredictResponse {
  case_id: number;
}

export interface CaseListItem {
  id: number;
  patient_id: string;
  visit_date: string;
  eye_side: string;
}

export interface CaseRecord {
  id: number;
  patient_id: string;
  age: number;
  sex: string;
  diabetes_duration_years: number;
  eye_side: string;
  visit_date: string;
  notes: string;
}

export interface PredictionRecord {
  predicted_class_index: number;
  predicted_label: string;
  expected_grade: number;
  confidence: number;
  class_probabilities: number[];
  ensemble_members: string[];
  ensemble_member_weights: number[];
  threshold_vector: number[];
  ereg_graph: EregGraph;
  gradcam_artifacts: GradcamArtifact[];
  ai_explanation: ModelExplanation;
}

export interface GradcamArtifact {
  model_name: string;
  display_name: string;
  image_url: string;
}

export interface ModelExplanation {
  summary: string;
  focus_region: string;
  focus_pattern: string;
  confidence_reason: string;
  class_margin: number;
}

export interface EregGraphNode {
  id: string;
  label: string;
  kind: string;
  detail: string;
  weight?: number | null;
}

export interface EregGraphEdge {
  source: string;
  target: string;
  label: string;
}

export interface EregGraph {
  nodes: EregGraphNode[];
  edges: EregGraphEdge[];
  steps: string[];
  summary: string;
  metrics: Record<string, number>;
}

export interface CaseDetailResponse {
  case: CaseRecord;
  prediction: PredictionRecord;
  original_image_url: string;
}
