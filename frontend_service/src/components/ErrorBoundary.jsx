import React from 'react';

/**
 * ErrorBoundary — F2
 * Catches any render-time React error below this component and shows a
 * user-friendly fallback instead of a blank white screen.
 */
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, info) {
        console.error('[ErrorBoundary]', error, info);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                    justifyContent: 'center', height: '100vh',
                    background: '#0a0f1e', color: '#e2e8f0', fontFamily: 'sans-serif'
                }}>
                    <div style={{ fontSize: '3rem', marginBottom: '16px' }}>⚠️</div>
                    <h2 style={{ color: '#f87171', marginBottom: '8px' }}>Something went wrong</h2>
                    <p style={{ color: '#94a3b8', maxWidth: '400px', textAlign: 'center' }}>
                        An unexpected error occurred. Please refresh the page.
                    </p>
                    <details style={{ marginTop: '20px', color: '#64748b', fontSize: '0.8rem' }}>
                        <summary style={{ cursor: 'pointer' }}>Error details</summary>
                        <pre style={{ marginTop: '8px', textAlign: 'left' }}>
                            {this.state.error?.toString()}
                        </pre>
                    </details>
                    <button
                        onClick={() => window.location.reload()}
                        style={{
                            marginTop: '24px', padding: '10px 24px',
                            background: 'rgba(0,242,255,0.15)',
                            border: '1px solid rgba(0,242,255,0.3)',
                            color: '#00f2ff', borderRadius: '8px',
                            cursor: 'pointer', fontSize: '0.95rem'
                        }}
                    >
                        Reload Page
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}

export default ErrorBoundary;
