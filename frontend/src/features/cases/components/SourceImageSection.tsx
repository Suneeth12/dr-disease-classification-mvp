import { apiUrl } from '../../../api/client';

interface SourceImageSectionProps {
  originalImageUrl: string;
  patientId: string;
}

export function SourceImageSection({ originalImageUrl, patientId }: SourceImageSectionProps) {
  return (
    <section className="panel source-image-panel" aria-labelledby="source-image-title">
      <div className="source-image-frame">
        <img
          src={apiUrl(originalImageUrl)}
          alt={`Fundus image for ${patientId}`}
          decoding="async"
          loading="lazy"
        />
      </div>
      <div>
        <p className="page-kicker">Original upload</p>
        <h2 className="section-title" id="source-image-title">Source image</h2>
        <p className="page-copy">
          This is the uploaded image served by the API and used for the prediction above.
        </p>
      </div>
    </section>
  );
}
