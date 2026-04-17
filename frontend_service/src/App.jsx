import React, { useState, useEffect, useRef } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, Title, Filler } from 'chart.js';
import { Doughnut, Line } from 'react-chartjs-2';
import axios from 'axios';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, Title, Filler);

const API_BASE = 'http://localhost:8001';
const WS_BASE = 'ws://localhost:8001';
const API_KEY = 'dev-secret-key'; // Matches INTERNAL_API_KEY in docker-compose

// --- Constants ---
const EMOTION_COLORS = {
    'joy': '#00ff88', 'happy': '#00ff88', 'love': '#ff66b2', 'admiration': '#00ffcc',
    'anger': '#ff0055', 'annoyance': '#ff6600', 'disgust': '#ff00aa',
    'sadness': '#0066ff', 'remorse': '#5500ff',
    'fear': '#7000ff', 'nervousness': '#9900ff',
    'surprise': '#ff9900', 'curiosity': '#ffcc00',
    'neutral': '#00f2ff'
};

const EMOTIONAL_WORDS = {
    'love': 'joy', 'happy': 'joy', 'great': 'joy', 'joy': 'joy', 'fun': 'joy',
    'hate': 'anger', 'angry': 'anger', 'mad': 'anger', 'stupid': 'anger', 'hell': 'anger',
    'kill': 'anger',
    'sad': 'sadness', 'cry': 'sadness', 'grief': 'sadness', 'sorry': 'sadness', 'regret': 'sadness',
    'fear': 'fear', 'scared': 'fear', 'afraid': 'fear', 'worry': 'fear',
    'wow': 'surprise', 'shock': 'surprise', 'amazing': 'surprise'
};

// --- Sub-components ---

// Word Highlighter Component
const EmotionalTextOverlay = ({ text }) => {
    if (!text) return null;
    const words = text.split(/\s+/);
    return (
        <div className="text-overlay" style={{ fontSize: '1.2rem', lineHeight: '1.6', marginBottom: '20px' }}>
            {words.map((word, i) => {
                const clean = word.toLowerCase().replace(/[^a-z]/g, '');
                const emo = EMOTIONAL_WORDS[clean];
                if (emo) {
                    const style = {
                        color: EMOTION_COLORS[emo] || 'white',
                        textShadow: `0 0 10px ${EMOTION_COLORS[emo]}`,
                        fontWeight: 'bold',
                        marginRight: '5px',
                        cursor: 'help'
                    };
                    return <span key={i} style={style} title={`${emo} impact detected`}>{word} </span>;
                }
                return <span key={i} style={{ marginRight: '5px', color: '#e2e8f0' }}>{word} </span>;
            })}
        </div>
    );
};

// Simple Plutchik Wheel Visualization (SVG)
const PlutchikWheel = ({ dominantEmotion }) => {
    const defaultColor = 'rgba(255,255,255,0.05)';
    const activeColor = 'var(--accent-primary)';

    // Mapping dominant emotions to Plutchik cores
    const mapping = {
        "admiration": "trust", "amusement": "joy", "approval": "trust", "caring": "trust",
        "desire": "anticipation", "excitement": "joy", "gratitude": "joy", "joy": "joy",
        "love": "joy", "optimism": "anticipation", "pride": "joy", "realization": "surprise",
        "relief": "joy", "surprise": "surprise", "curiosity": "surprise", "confusion": "surprise",
        "fear": "fear", "nervousness": "fear", "remorse": "sadness", "sadness": "sadness",
        "disappointment": "sadness", "grief": "sadness", "anger": "anger", "annoyance": "anger",
        "disapproval": "disgust", "disgust": "disgust", "embarrassment": "fear",
        "neutral": "neutral"
    };

    const core = mapping[dominantEmotion?.toLowerCase()] || "neutral";

    const petals = [
        { name: 'joy', rotate: 0 },
        { name: 'trust', rotate: 45 },
        { name: 'fear', rotate: 90 },
        { name: 'surprise', rotate: 135 },
        { name: 'sadness', rotate: 180 },
        { name: 'disgust', rotate: 225 },
        { name: 'anger', rotate: 270 },
        { name: 'anticipation', rotate: 315 },
    ];

    return (
        <div className="plutchik-container">
            <svg id="plutchik-svg" viewBox="0 0 200 200">
                <g transform="translate(100,100)">
                    {petals.map((petal, i) => (
                        <path
                            key={i}
                            d="M0,0 Q20,-40 0,-80 Q-20,-40 0,0"
                            fill={petal.name === core ? activeColor : defaultColor}
                            stroke="var(--glass-border)"
                            transform={`rotate(${petal.rotate})`}
                        />
                    ))}
                </g>
                <circle cx="100" cy="100" r="10" fill="white" filter="blur(2px)" />
            </svg>
        </div>
    );
};

const BuildupChart = ({ steps }) => {
    const labels = steps.map(s => s.text.length > 20 ? "..." + s.text.slice(-18) : s.text);
    const dataPoints = steps.map(s => {
        // Find top score
        const top = s.scores && s.scores.length > 0 ? Math.max(...s.scores.map(x => x.score)) : 0;
        return top;
    });

    // Attempt to map point colors
    const pointColors = steps.map(s => {
        const emo = s.dominant?.toLowerCase();
        return EMOTION_COLORS[emo] || '#00f2ff';
    });

    const data = {
        labels,
        datasets: [{
            label: 'Emotional Intensity',
            data: dataPoints,
            borderColor: '#00f2ff',
            backgroundColor: 'rgba(0, 242, 255, 0.1)',
            fill: true,
            tension: 0.4,
            pointBackgroundColor: pointColors,
            pointRadius: 6
        }]
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
            x: { grid: { display: false }, ticks: { display: false } } // Hide X ticks for cleaner look in small view
        },
        plugins: {
            legend: { display: false },
        }
    };

    return <Line data={data} options={options} />;
};


function App() {
    const [view, setView] = useState('live'); // 'live' | 'history' | 'analytics'
    const [status, setStatus] = useState('Live'); // 'Live' | 'Offline'
    const [messages, setMessages] = useState([]); // Chat history
    const [inputValue, setInputValue] = useState('');
    const [userId, setUserId] = useState('User 1');
    const [showEmojiPicker, setShowEmojiPicker] = useState(false);
    const [currentAnalysis, setCurrentAnalysis] = useState(null);
    const [vibeAnalysis, setVibeAnalysis] = useState(null); // Separate vibe state
    const [analyticsData, setAnalyticsData] = useState(null); // Calibration state

    // WebSocket
    const socketRef = useRef(null);
    const chatContainerRef = useRef(null);
    const messagesEndRef = useRef(null);

    // Initial Helper
    const applyTheme = (emotion) => {
        const e = emotion?.toLowerCase();
        const color = EMOTION_COLORS[e] || '#00f2ff';
        document.documentElement.style.setProperty('--accent-primary', color);
    };

    // Helper: Parse Analysis
    const parseAnalysis = (msg) => {
        if (!msg.emotions) return null;
        try {
            const ems = typeof msg.emotions === 'string' ? JSON.parse(msg.emotions) : msg.emotions;

            // Adapt to WS payload structure
            const bert_list = [];
            for (const [k, v] of Object.entries(ems)) {
                if (!['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'dominant_emotion', 'sentiment_positive', 'sentiment_negative'].includes(k)) {
                    bert_list.push({ label: k, score: v });
                }
            }

            return {
                type: 'analysis',
                data: {
                    id: msg.id,
                    raw_text: msg.content || msg.text,
                    final_dominant_emotion: ems.dominant_emotion || "Neutral",
                    final_valence: ems.vader_compound || 0,
                    bert_emotions: bert_list,
                    llm_insights: "Detailed analysis loaded.",
                    llm_sarcasm_score: 0,
                    hierarchical_scores: [],
                    emojis_found: [],
                    slang_detected: {}
                }
            };
        } catch (e) {
            console.error("Parse Error", e);
            return null;
        }
    };

    const fetchVibe = async () => {
        try {
            const stateRes = await axios.get(`${API_BASE}/conversation/conv-1/state`, {
                headers: { 'X-API-Key': API_KEY }
            });
            if (stateRes.data) {
                setVibeAnalysis({
                    valence: parseFloat(stateRes.data.average_valence || 0),
                    sync_score: 0.8,
                    resonance: 0.7,
                    volatility: 0.2,
                    top_emotions: [stateRes.data.dominant_emotion || "Neutral"]
                });
            }
        } catch (e) { console.warn("Could not fetch latest vibe state"); }
    };

    const fetchAnalytics = async () => {
        try {
            const res = await axios.get(`${API_BASE}/analytics/calibration`, {
                headers: { 'X-API-Key': API_KEY }
            });
            setAnalyticsData(res.data);
        } catch (e) { console.error("Failed to fetch analytics", e); }
    };

    // WebSocket Connection
    useEffect(() => {
        if (view === 'analytics') {
            fetchAnalytics();
        }
    }, [view]);
    useEffect(() => {
        // Fetch initial state first
        const fetchInitialState = async () => {
            try {
                // Fetch last 50 messages for history
                const res = await axios.get(`${API_BASE}/conversation/conv-1/messages?limit=50`, {
                    headers: { 'X-API-Key': API_KEY }
                });
                if (res.data && res.data.length > 0) {
                    // 1. Populate Chat History
                    const historyMsgs = res.data.slice().reverse().map(m => {
                        const parsed = parseAnalysis(m);
                        const isSelf = m.sender_id === 'User 1' || m.sender_id === 'User 2' ? (m.sender_id === userId) : false;

                        return {
                            id: m.id,
                            sender: isSelf ? 'user' : 'ai', // Visual distinction
                            text: m.content,
                            senderName: m.sender_id || 'System',
                            analysis: parsed
                        };
                    });
                    setMessages(historyMsgs);

                    // 2. Set Analysis Dashboard to the LATEST message's analysis
                    const lastMsg = res.data[0];
                    const initialAnalysis = parseAnalysis(lastMsg);
                    if (initialAnalysis) {
                        setCurrentAnalysis(initialAnalysis);
                        applyTheme(initialAnalysis.data.final_dominant_emotion);
                    }

                    // 3. Fetch Vibe
                    await fetchVibe();
                }
            } catch (err) {
                console.error("Initial fetch failed:", err);
            }
        };
        fetchInitialState();

        const clientId = `client_${Math.floor(Math.random() * 9999)}`;
        const ws = new WebSocket(`${WS_BASE}/ws/${clientId}?api_key=${API_KEY}`);

        ws.onopen = () => {
            console.log("Connected to WS");
            setStatus('Live');
        };

        ws.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload.type === 'analysis') {
                    setCurrentAnalysis(prev => ({
                        ...payload,
                        ai_insight: null,
                        loadingReasoning: true
                    }));
                    applyTheme(payload.data.final_dominant_emotion);

                    // Also update vibe if present
                    if (payload.vibe) {
                        setVibeAnalysis(payload.vibe);
                    } else {
                        // Force fetch new vibe
                        fetchVibe();
                    }
                } else if (payload.type === 'reasoning') {
                    setCurrentAnalysis(prev => {
                        if (prev && prev.data.id === payload.message_id) {
                            return { ...prev, ai_insight: payload.ai_insight, loadingReasoning: false };
                        }
                        return prev;
                    });
                }
            } catch (e) {
                console.error("WS Parse error", e);
            }
        };

        ws.onclose = () => setStatus('Offline');

        socketRef.current = ws;

        return () => ws.close();
    }, []); // Run once on mount. Changing userId doesn't reconnect WS, just changes sent ID.

    const sendMessage = () => {
        if (!inputValue.trim()) return;

        // Optimistic UI update
        const newMsg = {
            id: Date.now(),
            sender: 'user',
            text: inputValue,
            senderName: userId,
            analysis: null
        };
        setMessages(prev => [...prev, newMsg]);

        // Send via WS
        if (socketRef.current) {
            // Send explicit sender_id
            socketRef.current.send(JSON.stringify({
                text: inputValue,
                recipient_id: 'system',
                sender_id: userId
            }));
        }

        setInputValue('');
        setView('live');
    };

    // Auto-scroll chat: robust
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, view]);


    const handleFeedback = async (msgId, newLabel) => {
        try {
            await axios.post(`${API_BASE}/message/${msgId}/feedback`, 
                { label: newLabel },
                { headers: { 'X-API-Key': API_KEY } }
            );
            // Optionally update local state
            setMessages(prev => prev.map(m => m.id === msgId ? { ...m, verified: true, feedbackLabel: newLabel } : m));
            if (currentAnalysis && currentAnalysis.data.id === msgId) {
                setCurrentAnalysis(prev => ({ ...prev, verified: true, feedbackLabel: newLabel }));
            }
            console.log(`Feedback sent for ${msgId}: ${newLabel}`);
        } catch (e) {
            console.error("Failed to send feedback", e);
        }
    };
 
    const handleHistoryClick = (msg) => {
        if (msg.analysis) {
            setCurrentAnalysis(msg.analysis);
            applyTheme(msg.analysis.data.final_dominant_emotion);
            setView('live'); // Switch to analysis view
        } else {
            console.log("No analysis data for this message yet", msg);
        }
    };


    // --- Render Helpers ---

    return (
        <div className="app-shell">
            {/* Background */}
            <div className="bg-gradient"></div>
            <div className="glow-orb" id="orb-1"></div>
            <div className="glow-orb" id="orb-2"></div>

            {/* Sidebar */}
            <aside className="sidebar glass">
                <header className="app-header">
                    <div className="logo">
                        <div className="logo-icon"></div>
                        <h1>InnerLink</h1>
                    </div>
                    <div className="user-selector">
                        <select id="user-id-select" value={userId} onChange={e => setUserId(e.target.value)}>
                            <option value="User 1">User 1</option>
                            <option value="User 2">User 2</option>
                        </select>
                    </div>
                    <div className="status-indicator">
                        <span className="pulse"></span> {status}
                    </div>
                </header>

                <div className="nav-links">
                    <button className={`nav-btn ${view === 'live' ? 'active' : ''}`} onClick={() => setView('live')}>
                        <span className="btn-icon">⚡</span> Live Analysis
                    </button>
                    <button className={`nav-btn ${view === 'history' ? 'active' : ''}`} onClick={() => setView('history')}>
                        <span className="btn-icon">📜</span> History & Trends
                    </button>
                    <button className={`nav-btn ${view === 'analytics' ? 'active' : ''}`} onClick={() => setView('analytics')}>
                        <span className="btn-icon">📊</span> System Insights
                    </button>
                </div>

                <div className="chat-container" ref={chatContainerRef}>
                    {messages.length === 0 && (
                        <div className="empty-state">
                            <div className="empty-icon">✨</div>
                            <p>Enter a message to reveal its emotional subtext</p>
                        </div>
                    )}
                    {messages.map(msg => (
                        <div
                            key={msg.id}
                            className={`chat-msg ${msg.sender}`}
                            data-sender={msg.senderName}
                            onClick={() => handleHistoryClick(msg)}
                            style={{ cursor: msg.analysis ? 'pointer' : 'default', opacity: msg.analysis ? 1 : 0.8 }}
                            title={msg.analysis ? "Click to view emotional analysis" : "Analysis pending..."}
                        >
                            {msg.text}
                        </div>
                    ))}
                    <div ref={messagesEndRef}></div>
                </div>

                <div className="input-wrapper glass" style={{ position: 'relative' }}>
                    {showEmojiPicker && (
                        <div className="emoji-popup glass" style={{
                            position: 'absolute',
                            bottom: '100%',
                            left: '0',
                            marginBottom: '10px',
                            padding: '12px',
                            display: 'grid',
                            gridTemplateColumns: 'repeat(6, 1fr)',
                            gap: '8px',
                            zIndex: 1000,
                            background: 'rgba(17, 25, 40, 0.95)',
                            boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)'
                        }}>
                            {['😂', '😭', '😍', '🔥', '💀', '🙃', '🤔', '🙄', '💩', '❤️', '✨'].map(emoji => (
                                <button
                                    key={emoji}
                                    onClick={() => {
                                        setInputValue(prev => prev + emoji);
                                        setShowEmojiPicker(false);
                                    }}
                                    style={{
                                        background: 'none',
                                        border: 'none',
                                        fontSize: '1.5rem',
                                        cursor: 'pointer',
                                        transition: 'transform 0.1s'
                                    }}
                                    onMouseOver={e => e.target.style.transform = 'scale(1.2)'}
                                    onMouseOut={e => e.target.style.transform = 'scale(1)'}
                                >
                                    {emoji}
                                </button>
                            ))}
                        </div>
                    )}

                    <button
                        onClick={() => setShowEmojiPicker(!showEmojiPicker)}
                        style={{
                            background: 'none',
                            border: 'none',
                            fontSize: '1.5rem',
                            cursor: 'pointer',
                            opacity: 0.8,
                            padding: '0 8px',
                            display: 'flex',
                            alignItems: 'center',
                            transition: 'transform 0.2s'
                        }}
                        onMouseOver={e => e.target.style.transform = 'scale(1.1)'}
                        onMouseOut={e => e.target.style.transform = 'scale(1)'}
                        title="Quick Emojis"
                    >
                        😊
                    </button>
                    <textarea
                        rows="1"
                        placeholder="Type something expressive..."
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                    />
                    <button className="send-btn" onClick={sendMessage}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="22" y1="2" x2="11" y2="13"></line>
                            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                        </svg>
                    </button>
                </div>
            </aside>

            {/* Main Dashboard */}
            <main className="dashboard">
                <div className="dashboard-header">
                    <h2>Discovery Module</h2>
                    <div className="analysis-meta">
                        {currentAnalysis ? `ID: ${currentAnalysis.data.id.split('-')[0]}` : 'Ready'}
                    </div>
                </div>

                {/* IDLE VIEW */}
                {!currentAnalysis && view === 'live' && (
                    <div className="idle-view">
                        <div className="visualizer-mock">
                            <div className="bar"></div>
                            <div className="bar"></div>
                            <div className="bar"></div>
                            <div className="bar"></div>
                            <div className="bar"></div>
                        </div>
                        <p>Awaiting Signal Transmission</p>
                    </div>
                )}

                {/* LIVE ANALYSIS VIEW */}
                {currentAnalysis && view === 'live' && (
                    <div className="analysis-view">
                        {/* Vibe Dashboard */}
                        <section className="vibe-dashboard glass">
                            <div className="vibe-header">
                                <div className="vibe-title">
                                    <span className="vibe-icon">🌐</span>
                                    <h4>Conversation Vibe</h4>
                                </div>
                                <div className="vibe-status" style={{ color: 'var(--accent-success)' }}>
                                    Active
                                </div>
                            </div>
                            {vibeAnalysis ? (
                                <div className="vibe-body">
                                    <div className="vibe-metric">
                                        <label>Collective Valence</label>
                                        <div className="vibe-gauge">
                                            <div className="vibe-fill" style={{ width: `${((vibeAnalysis.valence + 1) / 2) * 100}%` }}></div>
                                        </div>
                                    </div>
                                    <div className="vibe-metric">
                                        <label>Sync Score</label>
                                        <div className="vibe-gauge">
                                            <div className="vibe-fill" style={{ width: `${vibeAnalysis.sync_score * 100}%` }}></div>
                                        </div>
                                    </div>
                                    <div className="vibe-metric">
                                        <label>Resonance</label>
                                        <div className="vibe-gauge">
                                            <div className="vibe-fill" style={{ width: `${vibeAnalysis.resonance * 100}%`, background: 'var(--accent-success)' }}></div>
                                        </div>
                                    </div>
                                    <div className="vibe-metric">
                                        <label>Volatility</label>
                                        <div className="vibe-gauge">
                                            <div className="vibe-fill" style={{ width: `${vibeAnalysis.volatility * 100}%`, background: 'var(--accent-error)' }}></div>
                                        </div>
                                    </div>
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

                        {/* Dominant Emotion */}
                        <section className="hero-emotion glass">
                            <div className="hero-emotion">
                                <span className="label">Dominant Resonance</span>
                                <h3 id="dominant-emotion-text" style={{ textShadow: `0 0 20px var(--accent-primary)` }}>
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
                                        onChange={(e) => handleFeedback(currentAnalysis.data.id, e.target.value)}
                                        value={currentAnalysis.feedbackLabel || ""}
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
 
                            {/* Word Highlighting Overlay */}
                            <EmotionalTextOverlay text={currentAnalysis.data.raw_text} />

                            <div className="valence-container">
                                <div className="valence-labels">
                                    <span>Negative</span>
                                    <span>Positive</span>
                                </div>
                                <div className="valence-track">
                                    <div className="valence-cursor" style={{ left: `${((currentAnalysis.data.final_valence + 1) / 2) * 100}%` }}></div>
                                </div>
                            </div>
                        </section>

                        <div className="detailed-grid">
                            {/* BERT Breakdown */}
                            <section className="analysis-card glass">
                                <div className="card-header">
                                    <h4>Neural Breakdown (BERT)</h4>
                                    <span className="info-tag">Confidence</span>
                                </div>
                                <div className="stats-list">
                                    {currentAnalysis.data.bert_emotions.sort((a, b) => b.score - a.score).slice(0, 4).map((item, i) => (
                                        <div key={i} className="stat-item">
                                            <div className="stat-header">
                                                <span>{item.label}</span>
                                                <span>{Math.round(item.score * 100)}%</span>
                                            </div>
                                            <div className="stat-bar-bg">
                                                <div className="stat-bar-fill" style={{ width: `${item.score * 100}%`, background: EMOTION_COLORS[item.label] || 'var(--accent-primary)' }}></div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </section>

                            {/* Dissonance / Sarcasm Meter (New) */}
                        {currentAnalysis.data.sarcasm_score > 0.1 && (
                            <div className={`dissonance-meter glass ${currentAnalysis.data.sarcasm_score > 0.5 ? 'high-alert' : ''}`}>
                                <div className="meter-label">
                                    <span>Contextual Dissonance</span>
                                    <span className="score">{Math.round(currentAnalysis.data.sarcasm_score * 100)}%</span>
                                </div>
                                <div className="meter-track">
                                    <div 
                                        className="meter-fill" 
                                        style={{ width: `${currentAnalysis.data.sarcasm_score * 100}%` }}
                                    ></div>
                                </div>
                                {currentAnalysis.data.conflict && (
                                    <p className="conflict-tag bounce-in">{currentAnalysis.data.conflict}</p>
                                )}
                            </div>
                        )}

                        {/* AI Insight Card */}
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
                                            <div className="dot"></div>
                                            <div className="dot"></div>
                                            <div className="dot"></div>
                                            <span>Deep analysis in progress...</span>
                                        </div>
                                    )}
                                </div>
                            </div>

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
                                                         style={{ width: `${impact * 100}%` }}>
                                                    </div>
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
                                    {/* Mocking historical context for demo. Ideally this comes from history. */}
                                    <BuildupChart steps={messages.filter(m => m.analysis).slice(-5).map(m => ({
                                        text: m.text,
                                        dominant: m.analysis.data.final_dominant_emotion,
                                        scores: [{ score: 0.8 }] // simplified
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
                )}

                {/* HISTORY VIEW */}
                {view === 'history' && (
                    <div className="history-view">
                        <h3>Conversation Archive</h3>
                        <div className="history-list">
                            {messages.map(msg => (
                                <div key={msg.id} className="history-item" onClick={() => handleHistoryClick(msg)} style={{ cursor: 'pointer' }}>
                                    <div className="history-item-header">
                                        <span>{msg.senderName}</span>
                                        <span>{new Date(msg.id).toLocaleTimeString()}</span>
                                    </div>
                                    <div className="history-item-text">
                                        {msg.text}
                                    </div>
                                    <div className="history-footer">
                                        {msg.analysis && (
                                            <div className="history-item-emotion" style={{ background: EMOTION_COLORS[msg.analysis.data.final_dominant_emotion.toLowerCase()] || '#888' }}>
                                                {msg.analysis.data.final_dominant_emotion}
                                            </div>
                                        )}
                                        <select 
                                            className="history-feedback-select"
                                            onClick={(e) => e.stopPropagation()}
                                            onChange={(e) => handleFeedback(msg.id, e.target.value)}
                                            value={msg.feedbackLabel || ""}
                                        >
                                            <option value="" disabled>✎</option>
                                            {Object.keys(EMOTION_COLORS).map(emo => (
                                                <option key={emo} value={emo}>{emo}</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
 
                {/* ANALYTICS VIEW */}
                {view === 'analytics' && (
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
                                {Object.entries(analyticsData?.emotion_breakdown || {}).sort((a,b) => b[1].samples - a[1].samples).map(([emo, stats]) => (
                                    <section key={emo} className="analysis-card glass">
                                        <div className="card-header">
                                            <h4>{emo}</h4>
                                            <span className="info-tag">{stats.samples} samples</span>
                                        </div>
                                        <div className="stats-list">
                                            <div className="stat-item">
                                                <div className="stat-header">
                                                    <span>Precision</span>
                                                    <span>{Math.round(stats.precision * 100)}%</span>
                                                </div>
                                                <div className="stat-bar-bg">
                                                    <div className="stat-bar-fill" style={{ width: `${stats.precision * 100}%`, background: 'var(--accent-primary)' }}></div>
                                                </div>
                                            </div>
                                            <div className="stat-item">
                                                <div className="stat-header">
                                                    <span>Recall</span>
                                                    <span>{Math.round(stats.recall * 100)}%</span>
                                                </div>
                                                <div className="stat-bar-bg">
                                                    <div className="stat-bar-fill" style={{ width: `${stats.recall * 100}%`, background: 'var(--accent-secondary)' }}></div>
                                                </div>
                                            </div>
                                        </div>
                                    </section>
                                ))}
                            </div>
                        )}
                    </div>
                )}
 
            </main>
        </div>
    );
}

export default App;
