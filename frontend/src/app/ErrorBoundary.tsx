import { Component, type ErrorInfo, type ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  errorMessage: string | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
    errorMessage: null,
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      errorMessage: error.message,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('App render error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <section className="app-error-panel panel animate-fade-in" aria-labelledby="app-error-title">
          <h2 id="app-error-title">Page failed to render</h2>
          <p className="page-copy">
            A runtime error interrupted the current page before React could finish rendering it.
          </p>
          <p className="app-error-message status-banner error" role="alert">
            {this.state.errorMessage ?? 'Unknown render error'}
          </p>
        </section>
      );
    }

    return this.props.children;
  }
}
