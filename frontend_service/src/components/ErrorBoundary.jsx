import React from 'react';

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
                <div className="flex flex-col items-center justify-center h-screen bg-[#0a0f1e] text-slate-200 font-sans">
                    <div className="text-5xl mb-4">⚠️</div>
                    <h2 className="text-red-400 mb-2">Something went wrong</h2>
                    <p className="text-slate-400 max-w-[400px] text-center">
                        An unexpected error occurred. Please refresh the page.
                    </p>
                    <details className="mt-5 text-slate-500 text-[0.8rem]">
                        <summary className="cursor-pointer">Error details</summary>
                        <pre className="mt-2 text-left">{this.state.error?.toString()}</pre>
                    </details>
                    <button
                        onClick={() => window.location.reload()}
                        className="mt-6 px-6 py-2.5 bg-[rgba(0,242,255,0.15)] border border-[rgba(0,242,255,0.3)] text-[#00f2ff] rounded-lg cursor-pointer text-[0.95rem]"
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
