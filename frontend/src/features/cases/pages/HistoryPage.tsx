import { useEffect, useState } from 'react';

import { listCases } from '../../../api/cases';
import type { CaseListItem } from '../../../api/types';
import { HistoryStatusPanel } from '../components/HistoryStatusPanel';
import { HistoryTable } from '../components/HistoryTable';

interface HistoryPageProps {
  onView: (id: number) => void;
}

function HistoryHeader({ copy }: { copy: string }) {
  return (
    <header className="history-hero page-header" aria-labelledby="history-page-title">
      <div>
        <p className="page-kicker">Case archive</p>
        <h1 className="history-hero__title page-title" id="history-page-title">
          Report archive
        </h1>
        <p className="history-hero__copy page-copy">{copy}</p>
      </div>
    </header>
  );
}

export default function HistoryPage({ onView }: HistoryPageProps) {
  const [cases, setCases] = useState<CaseListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadCases() {
      setLoading(true);
      setError(null);

      try {
        const payload = await listCases(controller.signal);
        setCases(payload);
      } catch (fetchError) {
        if (controller.signal.aborted) {
          return;
        }

        console.error(fetchError);
        setError(fetchError instanceof Error ? fetchError.message : 'Failed to load case history.');
        setCases([]);
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadCases();

    return () => controller.abort();
  }, []);

  if (loading) {
    return (
      <div className="history-page page-stack animate-fade-in" aria-busy="true">
        <HistoryHeader copy="Loading saved predictions and visit metadata." />
        <HistoryStatusPanel loading />
      </div>
    );
  }

  if (error) {
    return (
      <div className="history-page page-stack animate-fade-in">
        <HistoryHeader copy="Review previous predictions and reopen their evidence pages." />
        <HistoryStatusPanel error={error} />
      </div>
    );
  }

  return (
    <div className="history-page page-stack animate-fade-in">
      <HistoryHeader copy="Review saved predictions and reopen their full result pages." />

      <section className="history-archive-panel panel" aria-labelledby="history-archive-title">
        <div className="history-archive-header">
          <div>
            <p className="history-section-kicker">Saved reports</p>
            <h2 className="history-section-title" id="history-archive-title">
              Reviewed cases
            </h2>
          </div>
          <p className="history-record-count" aria-live="polite">
            {cases.length} {cases.length === 1 ? 'report' : 'reports'}
          </p>
        </div>

        <HistoryTable cases={cases} onView={onView} />
      </section>
    </div>
  );
}
