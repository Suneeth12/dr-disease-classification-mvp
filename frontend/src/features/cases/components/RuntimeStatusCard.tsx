import type { HealthResponse } from '../../../api/types';

interface RuntimeStatusCardProps {
  backendReady: boolean;
  backendMessage: string;
  health: HealthResponse | null;
}

export function RuntimeStatusCard({ backendReady, backendMessage, health }: RuntimeStatusCardProps) {
  const requiredArtifactEntries = health
    ? Object.entries(health.artifact_status).filter(([name]) => !name.startsWith('explainability::'))
    : [];
  const verifiedRequiredArtifactCount = requiredArtifactEntries.filter(([, passed]) => passed).length;
  const totalRequiredArtifactCount = requiredArtifactEntries.length;
  const computeDeviceLabel = health?.compute_device?.device_type
    ? `${health.compute_device.device_type} ${health.compute_device.device_name ?? ''}`.trim()
    : null;
  const runtimeCheckLabel = backendReady
    ? 'Models and thresholds ready'
    : totalRequiredArtifactCount > 0
      ? `Loading required artifacts (${verifiedRequiredArtifactCount}/${totalRequiredArtifactCount})`
      : 'Runtime status pending';

  return (
    <div className={`runtime-card ${backendReady ? 'ready' : 'loading'}`}>
      <span className="status-dot" aria-hidden="true" />
      <div>
        <strong>{backendReady ? 'Model runtime ready' : 'Models warming up'}</strong>
        <p>{backendMessage}</p>
        <small>{runtimeCheckLabel}</small>
        {computeDeviceLabel && <small>Device: {computeDeviceLabel}</small>}
      </div>
    </div>
  );
}
