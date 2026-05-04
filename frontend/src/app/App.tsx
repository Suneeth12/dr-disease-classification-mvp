import { useState } from 'react';

import HistoryPage from '../features/cases/pages/HistoryPage';
import NewCasePage from '../features/cases/pages/NewCasePage';
import ResultPage from '../features/cases/pages/ResultPage';
import { ErrorBoundary } from './ErrorBoundary';

type Page = 'newCase' | 'history' | 'result';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('newCase');
  const [currentResultId, setCurrentResultId] = useState<number | null>(null);

  const navigateToResult = (id: number) => {
    setCurrentResultId(id);
    setCurrentPage('result');
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block app-brand">
          <div className="brand-mark app-brand__mark" aria-hidden="true">
            DR
          </div>
          <div>
            <p className="brand-title app-brand__title">Dynamic DR Grading</p>
            <p className="brand-subtitle app-brand__subtitle">Classification, E-REG graph, and Grad-CAM</p>
          </div>
        </div>

        <nav className="nav-bar app-nav" aria-label="Primary navigation">
          <button
            type="button"
            className={`nav-link app-nav__item ${currentPage === 'newCase' ? 'active app-nav__item--active' : ''}`}
            aria-current={currentPage === 'newCase' ? 'page' : undefined}
            onClick={() => setCurrentPage('newCase')}
          >
            New Case
          </button>
          <button
            type="button"
            className={`nav-link app-nav__item ${currentPage === 'history' ? 'active app-nav__item--active' : ''}`}
            aria-current={currentPage === 'history' ? 'page' : undefined}
            onClick={() => setCurrentPage('history')}
          >
            History
          </button>
        </nav>
      </header>

      <main className="workspace app-workspace" id="main-content" tabIndex={-1}>
        <ErrorBoundary>
          {currentPage === 'newCase' && <NewCasePage onComplete={navigateToResult} />}
          {currentPage === 'history' && <HistoryPage onView={navigateToResult} />}
          {currentPage === 'result' && <ResultPage caseId={currentResultId} onBack={() => setCurrentPage('history')} />}
        </ErrorBoundary>
      </main>
    </div>
  );
}

export default App;
