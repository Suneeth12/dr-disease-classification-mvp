import type { ChangeEvent, FormEvent } from 'react';
import { useEffect, useState } from 'react';

import { createPrediction, getHealth } from '../../../api/cases';
import type { HealthResponse } from '../../../api/types';
import { PatientDetailsFields } from '../components/PatientDetailsFields';
import { RuntimeStatusCard } from '../components/RuntimeStatusCard';
import { UploadPanel } from '../components/UploadPanel';

interface NewCasePageProps {
  onComplete: (caseId: number) => void;
}

function isSupportedImage(file: File): boolean {
  const fileName = file.name.toLowerCase();
  return fileName.endsWith('.png') || fileName.endsWith('.jpg') || fileName.endsWith('.jpeg');
}

export default function NewCasePage({ onComplete }: NewCasePageProps) {
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendReady, setBackendReady] = useState(false);
  const [backendMessage, setBackendMessage] = useState('Checking backend status...');
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return undefined;
    }

    const previewUrl = URL.createObjectURL(file);
    setPreview(previewUrl);

    return () => URL.revokeObjectURL(previewUrl);
  }, [file]);

  useEffect(() => {
    let isMounted = true;
    let timerId: number | null = null;

    async function pollHealth() {
      try {
        const payload = await getHealth();

        if (!isMounted) {
          return;
        }
        setHealth(payload);

        if (payload.ready) {
          setBackendReady(true);
          setBackendMessage(payload.message ?? 'Backend is ready.');
          return;
        }

        setBackendReady(false);
        setBackendMessage(payload.message ?? 'Backend is loading the prediction runtime.');
      } catch (healthError) {
        if (!isMounted) {
          return;
        }

        console.error(healthError);
        setBackendReady(false);
        setBackendMessage('Backend is unavailable. Start the API on port 8000 and wait until model loading finishes.');
      }

      timerId = window.setTimeout(() => {
        void pollHealth();
      }, 5000);
    }

    void pollHealth();

    return () => {
      isMounted = false;
      if (timerId !== null) {
        window.clearTimeout(timerId);
      }
    };
  }, []);

  const handleImageChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;

    if (!nextFile) {
      setFile(null);
      setError(null);
      return;
    }

    if (!isSupportedImage(nextFile)) {
      setFile(null);
      setError('Please upload a PNG, JPG, or JPEG fundus image.');
      event.target.value = '';
      return;
    }

    setFile(nextFile);
    setError(null);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!file) {
      setError('Please select a PNG, JPG, or JPEG image before submitting.');
      return;
    }
    if (!backendReady) {
      setError(backendMessage || 'Backend is still loading models. Please wait a little longer.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData(event.currentTarget);
    formData.append('file', file);

    try {
      const payload = await createPrediction(formData);
      onComplete(payload.case_id);
    } catch (submitError) {
      console.error(submitError);
      if (submitError instanceof TypeError) {
        setError('Backend is unavailable. Make sure the API is running and fully started.');
      } else {
        setError(submitError instanceof Error ? submitError.message : 'Failed to connect to the backend.');
      }
    } finally {
      setLoading(false);
    }
  };

  const errorMessageId = error ? 'new-case-error' : undefined;

  return (
    <div className="new-case-page animate-fade-in">
      <section className="intake-hero" aria-labelledby="new-review-title">
        <div className="hero-copy">
          <p className="page-kicker">New review</p>
          <h1 className="page-title" id="new-review-title">Retinal grading with Grad-CAM review</h1>
          <p className="page-copy">
            Upload a fundus image. The model returns the DR grade first, then prepares Grad-CAM evidence on the result page.
          </p>

          <div className="review-flow" aria-label="Review workflow">
            <span>Patient</span>
            <span>Image</span>
            <span>Prediction</span>
            <span>Grad-CAM</span>
          </div>
        </div>

        <RuntimeStatusCard backendReady={backendReady} backendMessage={backendMessage} health={health} />
      </section>

      {!backendReady && (
        <div className="status-banner info" role="status">
          {backendMessage}
        </div>
      )}

      {error && (
        <div className="status-banner error" role="alert" id="new-case-error">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="new-case-layout">
        <PatientDetailsFields />
        <UploadPanel
          backendReady={backendReady}
          errorMessageId={errorMessageId}
          file={file}
          loading={loading}
          onImageChange={handleImageChange}
          preview={preview}
        />
      </form>
    </div>
  );
}
