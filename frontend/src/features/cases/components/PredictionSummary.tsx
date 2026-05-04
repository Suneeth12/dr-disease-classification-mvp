import type { CaseRecord, PredictionRecord } from '../../../api/types';

interface PredictionSummaryProps {
  caseData: CaseRecord;
  prediction: PredictionRecord;
}

export function PredictionSummary({ caseData, prediction }: PredictionSummaryProps) {
  return (
    <>
      <article className="diagnosis-panel panel">
        <p className="diagnosis-label">Predicted grade</p>
        <h2 className="diagnosis-title" id="prediction-summary-title">{prediction.predicted_label}</h2>
        <div className="diagnosis-metrics" aria-label="Prediction summary">
          <div>
            <span>Confidence</span>
            <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
          </div>
          <div>
            <span>Expected grade</span>
            <strong>{prediction.expected_grade.toFixed(2)}</strong>
          </div>
          <div>
            <span>Class</span>
            <strong>{prediction.predicted_class_index}</strong>
          </div>
        </div>
      </article>

      <article className="metadata-panel panel" aria-labelledby="patient-metadata-title">
        <div className="panel-heading">
          <div>
            <p className="diagnosis-label">Case context</p>
            <h2 className="section-title" id="patient-metadata-title">Patient metadata</h2>
          </div>
        </div>
        <dl className="metadata-list">
          <div>
            <dt>Age</dt>
            <dd>{caseData.age}</dd>
          </div>
          <div>
            <dt>Sex</dt>
            <dd>{caseData.sex}</dd>
          </div>
          <div>
            <dt>Duration</dt>
            <dd>{caseData.diabetes_duration_years} yrs</dd>
          </div>
          <div>
            <dt>Eye</dt>
            <dd>{caseData.eye_side}</dd>
          </div>
          <div className="metadata-note">
            <dt>Notes</dt>
            <dd>{caseData.notes || 'No notes provided.'}</dd>
          </div>
        </dl>
      </article>
    </>
  );
}
