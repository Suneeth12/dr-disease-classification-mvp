export function PatientDetailsFields() {
  return (
    <section className="panel patient-panel" aria-labelledby="patient-details-title">
      <div className="panel-heading stacked">
        <div>
          <p className="diagnosis-label">Step 1</p>
          <h2 className="section-title" id="patient-details-title">Patient details</h2>
        </div>
        <p className="section-note">Use simple case metadata for tracking and presentation.</p>
      </div>

      <div className="form-grid">
        <div className="form-field full">
          <label className="label" htmlFor="patient-id">Patient ID</label>
          <input id="patient-id" name="patient_id" required className="input" placeholder="e.g. PAT-1234" />
        </div>

        <div className="form-field">
          <label className="label" htmlFor="patient-age">Age</label>
          <input id="patient-age" name="age" type="number" min="0" required className="input" />
        </div>

        <div className="form-field">
          <label className="label" htmlFor="patient-sex">Sex</label>
          <select id="patient-sex" name="sex" className="input" required>
            <option value="M">Male</option>
            <option value="F">Female</option>
            <option value="O">Other</option>
          </select>
        </div>

        <div className="form-field">
          <label className="label" htmlFor="diabetes-duration">Diabetes duration</label>
          <input id="diabetes-duration" name="diabetes_duration_years" type="number" min="0" required className="input" />
        </div>

        <div className="form-field">
          <label className="label" htmlFor="eye-side">Eye side</label>
          <select id="eye-side" name="eye_side" className="input" required>
            <option value="Left">Left</option>
            <option value="Right">Right</option>
          </select>
        </div>

        <div className="form-field full">
          <label className="label" htmlFor="clinical-notes">Clinical notes</label>
          <textarea id="clinical-notes" name="notes" rows={4} className="input" />
        </div>
      </div>
    </section>
  );
}
