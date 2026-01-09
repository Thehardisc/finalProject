import React, { useState } from 'react';

// Sleek bar for emotions
const EmotionBar = ({ label, score, color }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.7rem', marginBottom: '4px' }}>
        <span style={{ width: '60px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', opacity: 0.8, color: '#a1a1aa' }}>{label}</span>
        <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px' }}>
            <div style={{ width: `${score * 100}%`, height: '100%', background: color, borderRadius: '2px', boxShadow: `0 0 10px ${color}` }} />
        </div>
        <span style={{ width: '25px', textAlign: 'right', color: color, fontWeight: 'bold' }}>{(score * 100).toFixed(0)}%</span>
    </div>
);

const MessageItem = ({ message, isOwn }) => {
    const [expanded, setExpanded] = useState(false);

    // Parse emotions
    let emotions = null;
    try {
        if (message.emotions) {
            emotions = typeof message.emotions === 'string' ? JSON.parse(message.emotions) : message.emotions;
        }
    } catch (e) { }

    const isPending = !emotions;

    // Get primary stats
    const vader = emotions?.vader_compound || 0;

    // Verdict Logic
    let sst_verdict = "NEU";
    let verdictColor = "#94a3b8"; // Gray
    if (emotions) {
        const neutralScore = emotions.neutral || 0;
        if (neutralScore > 0.4) {
            sst_verdict = "NEU";
            verdictColor = "#94a3b8"; // Slate-400
        } else {
            const isPos = (emotions.sentiment_positive || 0) > (emotions.sentiment_negative || 0);
            sst_verdict = isPos ? "POS" : "NEG";
            verdictColor = isPos ? "#4ade80" : "#f87171"; // Green-400 or Red-400
        }
    }

    // Top 3 GoEmotions
    let goLabels = [];
    if (emotions) {
        goLabels = Object.keys(emotions).filter(k =>
            !k.startsWith('vader_') && !k.startsWith('sentiment_') && !k.startsWith('emphasis_')
        ).sort((a, b) => (emotions[b] || 0) - (emotions[a] || 0)).slice(0, 3);
    }

    return (
        <div style={{
            alignSelf: 'flex-end',
            maxWidth: '75%',
            marginBottom: '1rem',
            animation: 'fadeIn 0.3s ease-out'
        }}>

            {/* The Unified Bubble - Cyberpunk Style */}
            <div
                onClick={() => setExpanded(!expanded)}
                style={{
                    background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 100%)', // Indigo-950 to Indigo-900
                    border: '1px solid #4338ca', // Indigo-700
                    color: '#e0e7ff', // Indigo-100
                    borderRadius: '1.2rem 1.2rem 0 1.2rem',
                    cursor: 'pointer',
                    boxShadow: '0 4px 15px rgba(49, 46, 129, 0.4)',
                    position: 'relative',
                    backdropFilter: 'blur(10px)',
                    overflow: 'hidden'
                }}
            >
                {/* Message Content */}
                <div style={{ padding: '1rem 1.2rem' }}>
                    <div style={{ lineHeight: '1.5', fontSize: '1rem' }}>{message.content}</div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem', opacity: 0.6, fontSize: '0.7rem' }}>
                        <span>{new Date(message.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            {!isPending && (
                                <span style={{
                                    padding: '2px 6px', borderRadius: '10px',
                                    background: `rgba(0,0,0,0.3)`,
                                    color: verdictColor, fontWeight: 'bold', border: `1px solid ${verdictColor}`
                                }}>
                                    {sst_verdict}
                                </span>
                            )}
                            {isPending && <span style={{ animation: 'pulse 1.5s infinite' }}>⌛ Analyzing...</span>}
                        </div>
                    </div>
                </div>

                {/* Integrated Analysis Section (Collapsible?) - No glass panel, just darker bg */}
                {!isPending && (
                    <div style={{
                        background: 'rgba(0, 0, 0, 0.3)',
                        borderTop: '1px solid rgba(255,255,255,0.1)',
                        padding: '0.8rem 1.2rem',
                        fontSize: '0.75rem'
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span style={{ color: '#a1a1aa', letterSpacing: '0.05em', textTransform: 'uppercase', fontSize: '0.65rem' }}>Dominant Emotions</span>
                            <span style={{ fontSize: '1rem' }}>
                                {/* Add an emoji based on top emotion? */}
                                {goLabels[0] === 'admiration' && '😍'}
                                {goLabels[0] === 'joy' && '😄'}
                                {goLabels[0] === 'anger' && '😡'}
                            </span>
                        </div>

                        {goLabels.map(emo => (
                            <EmotionBar key={emo} label={emo.toUpperCase()} score={emotions[emo]} color="#818cf8" />
                        ))}

                        {/* Emphasis Badge */}
                        {emotions?.emphasis_caps > 0 && (
                            <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '4px', color: '#fde047', fontSize: '0.7rem', fontWeight: 'bold' }}>
                                <span>⚡</span> HIGH INTENSITY
                            </div>
                        )}
                    </div>
                )}
            </div>

            <style>{`
                @keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
                @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            `}</style>
        </div>
    );
};

export default MessageItem;
