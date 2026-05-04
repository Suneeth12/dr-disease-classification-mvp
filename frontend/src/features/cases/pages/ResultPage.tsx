import { useCallback, useEffect, useRef, useState } from 'react';

import {
  EMPTY_MODEL_EXPLANATION,
  EXPLAINABILITY_MODEL_NAME,
  generateCaseExplainability,
  getCaseDetail,
} from '../../../api/cases';
import type { CaseDetailResponse } from '../../../api/types';
import { GradcamEvidenceSection } from '../components/GradcamEvidenceSection';
import { PredictionSummary } from '../components/PredictionSummary';
import { ProbabilityList } from '../components/ProbabilityList';
import { SourceImageSection } from '../components/SourceImageSection';

interface ResultPageProps {
  caseId: number | null;
  onBack: () => void;
}

export default function ResultPage({ caseId, onBack }: ResultPageProps) {
  const [data, setData] = useState<CaseDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [explainabilityLoading, setExplainabilityLoading] = useState(false);
  const [explainabilityError, setExplainabilityError] = useState<string | null>(null);
  const explainabilityRequestedIds = useRef<Set<number>>(new Set());

  useEffect(() => {
    if (!caseId) {
      setData(null);
      setError('No case selected.');
      setLoading(false);
      return undefined;
    }

    const selectedCaseId = caseId;
    const controller = new AbortController();

    async function loadCase() {
      setLoading(true);
      setError(null);

      try {
        const payload = await getCaseDetail(selectedCaseId, controller.signal);
        setData(payload);
      } catch (fetchError) {
        if (controller.signal.aborted) {
          return;
        }

        console.error(fetchError);
        setError(fetchError instanceof Error ? fetchError.message : 'Failed to load case result.');
        setData(null);
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadCase();

    return () => controller.abort();
  }, [caseId]);

  const requestExplainability = useCallback(async () => {
    if (!caseId || explainabilityLoading) {
      return;
    }

    const selectedCaseId = caseId;
    setExplainabilityLoading(true);
    setExplainabilityError(null);

    try {
      const payload = await generateCaseExplainability(selectedCaseId);
      setData(payload);
    } catch (generationError) {
      console.error(generationError);
      setExplainabilityError(
        generationError instanceof Error
          ? generationError.message
          : 'Failed to generate explainability overlays.',
      );
    } finally {
      setExplainabilityLoading(false);
    }
  }, [caseId, explainabilityLoading]);

  useEffect(() => {
    if (!caseId || !data) {
      return;
    }
    if (data.prediction.gradcam_artifacts.some((artifact) => artifact.model_name === EXPLAINABILITY_MODEL_NAME)) {
      return;
    }
    if (explainabilityRequestedIds.current.has(caseId)) {
      return;
    }

    explainabilityRequestedIds.current.add(caseId);
    void requestExplainability();
  }, [caseId, data, requestExplainability]);

  if (loading) {
    return (
      <div className="page-stack animate-fade-in">
        <header className="page-header">
          <div>
            <p className="page-kicker">Case evidence</p>
            <h1 className="page-title">Loading result</h1>
            <p className="page-copy">Retrieving the saved prediction, Grad-CAM, and patient metadata.</p>
          </div>
        </header>
        <div className="panel loading-state" role="status">Loading results...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="page-stack animate-fade-in">
        <header className="page-header">
          <div>
            <p className="page-kicker">Case evidence</p>
            <h1 className="page-title">Case result</h1>
            <p className="page-copy">The selected case could not be opened.</p>
          </div>
          <button type="button" className="button secondary" onClick={onBack}>
            Back to history
          </button>
        </header>
        <div className="status-banner error" role="alert">{error ?? 'Case result is unavailable.'}</div>
      </div>
    );
  }

  const { case: caseData, prediction, original_image_url: originalImageUrl } = data;
  const classProbabilities = Array.isArray(prediction.class_probabilities) ? prediction.class_probabilities : [];
  const gradcamArtifacts = Array.isArray(prediction.gradcam_artifacts)
    ? prediction.gradcam_artifacts.filter((artifact) => artifact.model_name === EXPLAINABILITY_MODEL_NAME)
    : [];
  const modelExplanation = prediction.ai_explanation ?? EMPTY_MODEL_EXPLANATION;

  return (
    <div className="result-page animate-fade-in">
      <header className="result-topline" aria-labelledby="result-title">
        <div>
          <p className="page-kicker">Case evidence</p>
          <h1 className="page-title" id="result-title">Result for {caseData.patient_id}</h1>
          <p className="page-copy">
            {new Date(caseData.visit_date).toLocaleString()} · {caseData.eye_side} eye · age {caseData.age}
          </p>
        </div>
        <button type="button" className="button secondary" onClick={onBack}>
          Back to history
        </button>
      </header>

      <section className="report-overview" aria-labelledby="prediction-summary-title">
        <PredictionSummary caseData={caseData} prediction={prediction} />
        <ProbabilityList predictedClassIndex={prediction.predicted_class_index} probabilities={classProbabilities} />
      </section>

      <GradcamEvidenceSection
        caseData={caseData}
        eregGraph={prediction.ereg_graph}
        explainabilityError={explainabilityError}
        explainabilityLoading={explainabilityLoading}
        gradcamArtifacts={gradcamArtifacts}
        modelExplanation={modelExplanation}
        onGenerateExplainability={() => {
          void requestExplainability();
        }}
      />

      <SourceImageSection originalImageUrl={originalImageUrl} patientId={caseData.patient_id} />
    </div>
  );
}
