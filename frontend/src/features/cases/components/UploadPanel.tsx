import type { ChangeEvent } from 'react';

interface UploadPanelProps {
  backendReady: boolean;
  errorMessageId?: string;
  file: File | null;
  loading: boolean;
  onImageChange: (event: ChangeEvent<HTMLInputElement>) => void;
  preview: string | null;
}

export function UploadPanel({
  backendReady,
  errorMessageId,
  file,
  loading,
  onImageChange,
  preview,
}: UploadPanelProps) {
  return (
    <section className="panel upload-panel" aria-labelledby="fundus-image-title">
      <div className="panel-heading stacked">
        <div>
          <p className="diagnosis-label">Step 2</p>
          <h2 className="section-title" id="fundus-image-title">Fundus image</h2>
        </div>
        <p className="section-note">PNG or JPEG fundus image, previewed before prediction.</p>
      </div>

      <div className="upload-control">
        <input
          id="fundus-image"
          type="file"
          accept=".png,.jpg,.jpeg,image/png,image/jpeg"
          onChange={onImageChange}
          className="file-input"
          aria-describedby={errorMessageId ? `fundus-image-help ${errorMessageId}` : 'fundus-image-help'}
          aria-invalid={Boolean(errorMessageId && !file)}
        />
        <label className={`upload-dropzone ${preview ? 'has-preview' : ''}`} htmlFor="fundus-image">
          {preview ? (
            <img src={preview} alt="Selected fundus image preview" decoding="async" />
          ) : (
            <span className="upload-empty">
              <strong>Choose fundus image</strong>
              <small>PNG, JPG, or JPEG</small>
            </span>
          )}
        </label>
        <p className="upload-helper" id="fundus-image-help">
          Select a clear retinal fundus image before running the prediction.
        </p>
      </div>

      <div className="selected-file-row">
        <span>Selected file</span>
        <strong title={file?.name}>{file?.name ?? 'None'}</strong>
      </div>

      <button type="submit" disabled={loading || !backendReady} aria-busy={loading} className="button">
        {loading
          ? 'Running Fast Prediction...'
          : backendReady
            ? 'Run Prediction'
            : 'Backend is Warming Up...'}
      </button>

      <div className="submission-steps" aria-live="polite">
        <div className={`submission-step ${file ? 'complete' : ''}`}>
          <span>1</span>
          <p>Image selected</p>
        </div>
        <div className={`submission-step ${backendReady ? 'complete' : ''}`}>
          <span>2</span>
          <p>Models loaded</p>
        </div>
        <div className={`submission-step ${loading ? 'active' : ''}`}>
          <span>3</span>
          <p>Prediction saves case</p>
        </div>
      </div>
    </section>
  );
}
