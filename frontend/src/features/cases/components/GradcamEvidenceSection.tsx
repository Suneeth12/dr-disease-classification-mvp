import { apiUrl } from '../../../api/client';
import { EMPTY_MODEL_EXPLANATION } from '../../../api/cases';
import type { CaseRecord, EregGraph, GradcamArtifact, ModelExplanation } from '../../../api/types';

interface GradcamEvidenceSectionProps {
  caseData: CaseRecord;
  explainabilityError: string | null;
  explainabilityLoading: boolean;
  eregGraph: EregGraph;
  gradcamArtifacts: GradcamArtifact[];
  modelExplanation: ModelExplanation | null;
  onGenerateExplainability: () => void;
}

export function GradcamEvidenceSection({
  caseData,
  explainabilityError,
  explainabilityLoading,
  eregGraph,
  gradcamArtifacts,
  modelExplanation,
  onGenerateExplainability,
}: GradcamEvidenceSectionProps) {
  const selectedArtifact = gradcamArtifacts[0] ?? null;
  const explanation = modelExplanation ?? EMPTY_MODEL_EXPLANATION;
  const memberNodes = eregGraph.nodes.filter((node) => node.kind === 'member');
  const flowNodes = eregGraph.nodes.filter((node) => node.kind !== 'member');
  const graphContent = (
    <div className="ereg-graph" aria-label="E-REG graph explanation">
      <p className="ereg-graph__summary">{eregGraph.summary}</p>
      {memberNodes.length > 0 && (
        <div className="ereg-graph__members" aria-label="E-REG member model weights">
          {memberNodes.map((node) => (
            <div className="ereg-member" key={node.id}>
              <span>{node.label}</span>
              <strong>{node.weight == null ? 'n/a' : `${(node.weight * 100).toFixed(1)}%`}</strong>
            </div>
          ))}
        </div>
      )}
      {flowNodes.length > 0 && (
        <div className="ereg-graph__flow" aria-label="E-REG decision flow">
          {flowNodes.map((node) => (
            <div className={`ereg-graph-node ereg-graph-node--${node.kind}`} key={node.id}>
              <span>{node.kind}</span>
              <strong>{node.label}</strong>
              {node.detail && <small>{node.detail}</small>}
            </div>
          ))}
        </div>
      )}
      {eregGraph.steps.length > 0 && (
        <ol className="ereg-graph__steps" aria-label="E-REG explanation steps">
          {eregGraph.steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      )}
    </div>
  );

  return (
    <section className="panel explainability-panel" aria-labelledby="gradcam-evidence-title">
      <div className="evidence-header">
        <div>
          <p className="page-kicker">Explainability</p>
          <h2 className="section-title" id="gradcam-evidence-title">E-REG graph and Grad-CAM</h2>
          <p className="page-copy">
            The member models are explained through the E-REG decision graph. The visual heatmap is shown as Grad-CAM.
          </p>
        </div>
      </div>

      {gradcamArtifacts.length === 0 ? (
        <>
          <div className="explainability-empty">
            <div>
              <p className="diagnosis-label">Fast prediction mode</p>
              <h3 className="section-title">Diagnosis is ready. Grad-CAM is being prepared.</h3>
              <p className="page-copy">
                The system returns the prediction and E-REG graph first, then generates the Grad-CAM heatmap for review.
              </p>
              {explainabilityError && <div className="status-banner error" role="alert">{explainabilityError}</div>}
            </div>
            <button
              type="button"
              className="button"
              disabled={explainabilityLoading}
              aria-busy={explainabilityLoading}
              onClick={onGenerateExplainability}
            >
              {explainabilityLoading ? 'Generating Grad-CAM...' : 'Generate Grad-CAM'}
            </button>
          </div>
          {eregGraph.nodes.length > 0 && (
            <aside className="model-explanation-panel" aria-labelledby="ereg-fast-graph-title">
              <div>
                <p className="diagnosis-label" id="ereg-fast-graph-title">E-REG decision graph</p>
              </div>
              {graphContent}
            </aside>
          )}
        </>
      ) : (
        selectedArtifact && (
          <div className="explainability-grid">
            <figure className="gradcam-frame" aria-label="Grad-CAM heatmap">
              <img
                src={apiUrl(selectedArtifact.image_url)}
                alt={`Grad-CAM heatmap overlay for ${caseData.patient_id}`}
                decoding="async"
                loading="eager"
              />
            </figure>

            <aside className="model-explanation-panel" aria-labelledby="model-reasoning-title">
              <div>
                <p className="diagnosis-label" id="model-reasoning-title">E-REG decision graph</p>
                <p className="model-summary">{explanation.summary}</p>
              </div>

              <div className="reason-grid">
                <div>
                  <span>Primary focus</span>
                  <strong>{explanation.focus_region}</strong>
                </div>
                <div>
                  <span>Focus pattern</span>
                  <strong>{explanation.focus_pattern}</strong>
                </div>
                <div>
                  <span>Evidence map</span>
                  <strong>Grad-CAM</strong>
                </div>
                <div>
                  <span>Class margin</span>
                  <strong>{(explanation.class_margin * 100).toFixed(1)}%</strong>
                </div>
              </div>

              <p className="confidence-copy">{explanation.confidence_reason}</p>

              {graphContent}
            </aside>
          </div>
        )
      )}
    </section>
  );
}
