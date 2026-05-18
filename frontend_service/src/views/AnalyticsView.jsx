import React from 'react';

/**
 * AnalyticsView — Model calibration stats derived from verified feedback.
 */
const AnalyticsView = ({ analyticsData }) => (
    <div className="analytics-view">
        <section className="glass hero-analytics">
            <div className="analytics-summary">
                <h3>Model Calibration</h3>
                <p>Performance metrics derived from {analyticsData?.total_verified_samples || 0} verified samples.</p>
                {analyticsData?.overall_accuracy && (
                    <div className="main-accuracy">
                        <div className="accuracy-value">{Math.round(analyticsData.overall_accuracy * 100)}%</div>
                        <div className="label">Composite Accuracy</div>
                    </div>
                )}
            </div>
        </section>

        {analyticsData?.status === 'no_data' ? (
            <div className="idle-view glass">
                <p>Insufficient verification data. Use the feedback tools in History to calibrate the system.</p>
            </div>
        ) : (
            <div className="detailed-grid">
                {Object.entries(analyticsData?.emotion_breakdown || {})
                    .sort((a, b) => b[1].samples - a[1].samples)
                    .map(([emo, stats]) => (
                        <section key={emo} className="analysis-card glass">
                            <div className="card-header">
                                <h4>{emo}</h4>
                                <span className="info-tag">{stats.samples} samples</span>
                            </div>
                            <div className="stats-list">
                                {[
                                    { label: 'Precision', val: stats.precision, color: 'var(--accent-primary)' },
                                    { label: 'Recall',    val: stats.recall,    color: 'var(--accent-secondary)' }
                                ].map(({ label, val, color }) => (
                                    <div key={label} className="stat-item">
                                        <div className="stat-header">
                                            <span>{label}</span>
                                            <span>{Math.round(val * 100)}%</span>
                                        </div>
                                        <div className="stat-bar-bg">
                                            <div className="stat-bar-fill"
                                                 style={{ width: `${val * 100}%`, background: color }} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </section>
                    ))}
            </div>
        )}
    </div>
);

export default AnalyticsView;
