import React from 'react';
import { EMOTION_COLORS } from '../constants/emotions';
import EmotionalTextOverlay from '../visualizations/EmotionalTextOverlay';
import PlutchikWheel from '../visualizations/PlutchikWheel';
import BuildupChart from '../visualizations/BuildupChart';

/**
 * LiveView — Main analysis dashboard shown after a message is processed.
 */
const LiveView = ({ currentAnalysis, vibeAnalysis, messages, handleFeedback }) => {
    if (!currentAnalysis) {
        return (
            <div className="idle-view">
                <div className="visualizer-mock">
                    <div className="bar" /><div className="bar" /><div className="bar" />
                    <div className="bar" /><div className="bar" />
                </div>
                <p>Awaiting Signal Transmission</p>
            </div>
        );
    }

    return (
        <div className="analysis-view">
            {/* Vibe section */}
            <section className="vibe-dashboard glass">
                <div className="vibe-header">
                    <div className="vibe-title">
                        <span className="vibe-icon">🌐</span>
                        <h4>Conversation Vibe</h4>
                    </div>
                    <div className="vibe-status" style={{ color: 'var(--accent-success)' }}>Active</div>
                </div>
                {vibeAnalysis ? (
                    <div className="vibe-body">
                        {[
                            { label: 'Collective Valence', val: (vibeAnalysis.valence + 1) / 2 },
                            { label: 'Sync Score',         val: vibeAnalysis.sync_score },
                            { label: 'Resonance',          val: vibeAnalysis.resonance, color: 'var(--accent-success)' },
                            { label: 'Volatility',         val: vibeAnalysis.volatility, color: 'var(--accent-error)' }
                        ].map(({ label, val, color }) => (
                            <div key={label} className="vibe-metric">
                                <label>{label}</label>
                                <div className="vibe-gauge">
                                    <div className="vibe-fill"
                                         style={{ width: `${val * 100}%`, ...(color ? { background: color } : {}) }} />
                                </div>
                            </div>
                        ))}
                        <div className="vibe-top-emotions">
                            {vibeAnalysis.top_emotions.map((e, i) => (
                                <span key={i} className="vibe-tag">#{e}</span>
                            ))}
                        </div>
                    </div>
                ) : (
                    <div style={{ padding: '20px', color: '#64748b' }}>Loading Vibe State...</div>
                )}
            </section>

            {/* Dominant emotion */}
            <section className="hero-emotion glass">
                <div className="hero-emotion">
                    <span className="label">Dominant Resonance</span>
                    <h3 id="dominant-emotion-text" style={{ textShadow: '0 0 20px var(--accent-primary)' }}>
                        {currentAnalysis.data.final_dominant_emotion}
                    </h3>
                    {currentAnalysis.data.context_shift && (
                        <div className="context-shift-badge glass pulse-neon">
                            <span className="shift-icon">🔄</span>
                            <div className="shift-content">
                                <strong>Context Shift</strong>
                                <span>From {currentAnalysis.data.context_shift.from} to {currentAnalysis.data.context_shift.to}</span>
                            </div>
                        </div>
                    )}
                </div>

                <div className="feedback-section">
                    <span className="label">Is this correct?</span>
                    <div className="feedback-actions">
                        <select
                            className="glass feedback-select"
                            onChange={e => handleFeedback(currentAnalysis.data.id, e.target.value)}
                            value={currentAnalysis.feedbackLabel || ''}
                        >
                            <option value="" disabled>Correct this emotion...</option>
                            {Object.keys(EMOTION_COLORS).map(emo => (
                                <option key={emo} value={emo}>{emo}</option>
                            ))}
                        </select>
                        {(currentAnalysis.verified || currentAnalysis.feedbackLabel) && (
                            <span className="verified-badge">✓ Verified</span>
                        )}
                    </div>
                </div>

                <EmotionalTextOverlay text={currentAnalysis.data.raw_text} />

                <div className="valence-container">
                    <div className="valence-labels">
                        <span>Negative</span><span>Positive</span>
                    </div>
                    <div className="valence-track">
                        <div className="valence-cursor"
                             style={{ left: `${((currentAnalysis.data.final_valence + 1) / 2) * 100}%` }} />
                    </div>
                </div>
            </section>

            <div className="detailed-grid">
                {/* BERT breakdown */}
                <section className="analysis-card glass">
                    <div className="card-header">
                        <h4>Neural Breakdown (BERT)</h4>
                        <span className="info-tag">Confidence</span>
                    </div>
                    <div className="stats-list">
                        {currentAnalysis.data.bert_emotions
                            .sort((a, b) => b.score - a.score).slice(0, 4)
                            .map((item, i) => (
                                <div key={i} className="stat-item">
                                    <div className="stat-header">
                                        <span>{item.label}</span>
                                        <span>{Math.round(item.score * 100)}%</span>
                                    </div>
                                    <div className="stat-bar-bg">
                                        <div className="stat-bar-fill"
                                             style={{ width: `${item.score * 100}%`,
                                                      background: EMOTION_COLORS[item.label] || 'var(--accent-primary)' }} />
                                    </div>
                                </div>
                            ))}
                    </div>
                </section>

                {/* Sarcasm meter */}
                {currentAnalysis.data.sarcasm_score > 0.1 && (
                    <div className={`dissonance-meter glass ${currentAnalysis.data.sarcasm_score > 0.5 ? 'high-alert' : ''}`}>
                        <div className="meter-label">
                            <span>Contextual Dissonance</span>
                            <span className="score">{Math.round(currentAnalysis.data.sarcasm_score * 100)}%</span>
                        </div>
                        <div className="meter-track">
                            <div className="meter-fill"
                                 style={{ width: `${currentAnalysis.data.sarcasm_score * 100}%` }} />
                        </div>
                        {currentAnalysis.data.conflict && (
                            <p className="conflict-tag bounce-in">{currentAnalysis.data.conflict}</p>
                        )}
                    </div>
                )}

                {/* AI Insight */}
                <div className={`ai-insight-card glass ${currentAnalysis.loadingReasoning ? 'thinking' : ''}`}>
                    <div className="card-header">
                        <span className="icon">🧠</span>
                        <h4>AI Cognitive Insight</h4>
                    </div>
                    <div className="card-body">
                        {currentAnalysis.ai_insight ? (
                            <p className="insight-text fade-in">{currentAnalysis.ai_insight}</p>
                        ) : (
                            <div className="thinking-loader">
                                <div className="dot" /><div className="dot" /><div className="dot" />
                                <span>Deep analysis in progress...</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Logic map */}
                <div className="logic-explainer section">
                    <h4 className="section-title">Logic Map — Model Influence</h4>
                    {currentAnalysis.data.logic_map && Object.keys(currentAnalysis.data.logic_map).length > 0 ? (
                        <div className="impact-grid">
                            {Object.entries(currentAnalysis.data.logic_map).map(([system, impact]) => (
                                <div key={system} className="impact-item">
                                    <div className="impact-header">
                                        <span>{system}</span>
                                        <span>{Math.round(impact * 100)}%</span>
                                    </div>
                                    <div className="impact-bar-bg">
                                        <div className={`impact-bar-fill impact-${system.toLowerCase()}`}
                                             style={{ width: `${impact * 100}%` }} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="no-data-msg">Logic trace unavailable for this message.</p>
                    )}
                </div>

                {/* Buildup Chart */}
                <section className="analysis-card glass full-width">
                    <div className="card-header">
                        <h4>Emotional Resonance Progression</h4>
                        <span className="info-tag">Semantic Build-up</span>
                    </div>
                    <div className="chart-container-live">
                        <BuildupChart steps={messages.filter(m => m.analysis).slice(-5).map(m => ({
                            text:     m.text,
                            dominant: m.analysis.data.final_dominant_emotion,
                            scores:   [{ score: 0.8 }]
                        }))} />
                    </div>
                </section>

                {/* Plutchik */}
                <section className="analysis-card glass plutchik-card">
                    <div className="card-header">
                        <h4>Emotional Geometry (Plutchik)</h4>
                    </div>
                    <PlutchikWheel dominantEmotion={currentAnalysis.data.final_dominant_emotion} />
                </section>
            </div>
        </div>
    );
};

export default LiveView;
