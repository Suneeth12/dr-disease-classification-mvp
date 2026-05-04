interface HistoryStatusPanelProps {
  error?: string;
  loading?: boolean;
}

export function HistoryStatusPanel({ error, loading = false }: HistoryStatusPanelProps) {
  return (
    <section className={`history-status-panel panel ${loading ? 'loading-state' : ''}`} aria-label="History status">
      {error ? (
        <p className="history-error-message status-banner error" role="alert">
          {error}
        </p>
      ) : (
        <p className="history-status-message" role="status" aria-live="polite" aria-busy={loading}>
          Loading report archive...
        </p>
      )}
    </section>
  );
}
