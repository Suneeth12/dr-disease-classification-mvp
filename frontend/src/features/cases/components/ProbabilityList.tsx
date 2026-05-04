import { CLASS_LABELS } from '../../../api/cases';

interface ProbabilityListProps {
  predictedClassIndex: number;
  probabilities: number[];
}

export function ProbabilityList({ predictedClassIndex, probabilities }: ProbabilityListProps) {
  const strongestAlternative = probabilities.reduce(
    (best, probability, index) => {
      if (index === predictedClassIndex) {
        return best;
      }
      return probability > best.probability ? { index, probability } : best;
    },
    { index: -1, probability: -1 },
  );

  return (
    <article className="probability-panel panel" aria-labelledby="class-probabilities-title">
      <div className="panel-heading">
        <div>
          <p className="diagnosis-label">Distribution</p>
          <h2 className="section-title" id="class-probabilities-title">Class probabilities</h2>
        </div>
        {strongestAlternative.index >= 0 && (
          <p className="probability-note">
            Next: {CLASS_LABELS[strongestAlternative.index] ?? `Class ${strongestAlternative.index}`} {(strongestAlternative.probability * 100).toFixed(1)}%
          </p>
        )}
      </div>
      {probabilities.length === 0 ? (
        <div className="empty-state">No probability vector was returned for this case.</div>
      ) : (
        <div className="probability-list">
          {probabilities.map((probability, index) => (
            <div key={CLASS_LABELS[index] ?? `Class ${index}`} className="probability-item">
              <div className="probability-label">
                <span>{CLASS_LABELS[index] ?? `Class ${index}`}</span>
                <strong>{(probability * 100).toFixed(1)}%</strong>
              </div>
              <div className="probability-track">
                <div className="probability-fill" style={{ width: `${Math.max(0, Math.min(1, probability)) * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}
