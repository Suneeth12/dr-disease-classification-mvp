import type { CaseListItem } from '../../../api/types';

interface HistoryTableProps {
  cases: CaseListItem[];
  onView: (id: number) => void;
}

export function HistoryTable({ cases, onView }: HistoryTableProps) {
  if (cases.length === 0) {
    return (
      <div className="history-empty-state empty-state" role="status">
        <h3 className="history-empty-state__title">No reports saved yet</h3>
        <p className="history-empty-state__copy">
          Run a prediction to create the first archived report for this workspace.
        </p>
      </div>
    );
  }

  return (
    <div className="history-table-wrap">
      <table className="history-table" aria-describedby="history-table-summary">
        <caption className="history-table-caption">Saved retinal review reports</caption>
        <thead>
          <tr>
            <th scope="col">Patient ID</th>
            <th scope="col">Visit date</th>
            <th scope="col">Eye</th>
            <th scope="col">Action</th>
          </tr>
        </thead>
        <tbody>
          {cases.map((caseItem) => (
            <tr key={caseItem.id}>
              <th scope="row">{caseItem.patient_id}</th>
              <td>
                <time dateTime={caseItem.visit_date}>
                  {new Date(caseItem.visit_date).toLocaleString()}
                </time>
              </td>
              <td>{caseItem.eye_side}</td>
              <td>
                <button
                  type="button"
                  className="history-view-button button secondary"
                  aria-label={`View report for patient ${caseItem.patient_id}`}
                  onClick={() => onView(caseItem.id)}
                >
                  View result
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="history-table-summary" id="history-table-summary">
        Each row opens the saved result page with prediction details and supporting evidence.
      </p>
    </div>
  );
}
