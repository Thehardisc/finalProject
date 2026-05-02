import React, { useState, useEffect, useRef } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, Title, Filler } from 'chart.js';
import { Doughnut, Line } from 'react-chartjs-2';
import axios from 'axios';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, Title, Filler);

const API_BASE = 'http://localhost:8001';
const WS_BASE = 'ws://localhost:8001';
const API_KEY = 'il-9fA3mK7wXrPq2nZeVtYs'; // Matches INTERNAL_API_KEY in .env

// constants
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

// sub components

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

// Plutchik wheel
const PlutchikWheel = ({ dominantEmotion }) => {
    const defaultColor = 'rgba(255,255,255,0.05)';
    const activeColor = 'var(--accent-primary)';

    // map emotion to core
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

    // map point colors
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
    const [view, setView] = useState('live');
    const [status, setStatus] = useState('Offline');
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [currentUser, setCurrentUser] = useState(null);
    const [loginUsername, setLoginUsername] = useState('');
    const [conversations, setConversations] = useState([]);
    const [globalUsers, setGlobalUsers] = useState([]);
    const [activeConversationId, setActiveConversationId] = useState(null);
    const [activeTab, setActiveTab] = useState('chats');
    const [showEmojiPicker, setShowEmojiPicker] = useState(false);
    const [currentAnalysis, setCurrentAnalysis] = useState(null);
    const [vibeAnalysis, setVibeAnalysis] = useState(null);
    const [analyticsData, setAnalyticsData] = useState(null);
    const [systemReady, setSystemReady] = useState(false);
    const [componentStatus, setComponentStatus] = useState({
        database: false,
        redis: false,
        meta_learner: false,
    });
    const [gateVisible, setGateVisible] = useState(false); // Hidden until user logs in
    const [trainingInProgress, setTrainingInProgress] = useState(false); // Shows banner when model training

    // refs
    const socketRef = useRef(null);
    const chatContainerRef = useRef(null);
    const messagesEndRef = useRef(null);

    // helpers
    const applyTheme = (emotion) => {
        const e = emotion?.toLowerCase();
        const color = EMOTION_COLORS[e] || '#00f2ff';
        document.documentElement.style.setProperty('--accent-primary', color);
    };

    // parse message
    const parseAnalysis = (msg) => {
        if (!msg.emotions) return null;
        try {
            const ems = typeof msg.emotions === 'string' ? JSON.parse(msg.emotions) : msg.emotions;

            // transform data
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
        if (!activeConversationId) return;
        try {
            const stateRes = await axios.get(`${API_BASE}/conversation/${activeConversationId}/state`, {
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

    const checkSystemStatus = async () => {
        try {
            const res = await axios.get(`${API_BASE}/health/status`, {
                headers: { 'X-API-Key': API_KEY },
                validateStatus: (s) => s === 200 || s === 503, // Accept 503 as valid data
            });
            const data = res.data;
            if (data.components) {
                setComponentStatus(data.components);
            }
            if (data.ready) {
                // fade out and hide gate
                setGateVisible(false);
                setTimeout(() => setSystemReady(true), 800);
                // Check if trainer is still on its first cycle (model is old/pre-existing)
                setTrainingInProgress(data.training_in_progress === true);
                return true;
            } else {
                // Not ready yet — check if training is at least underway
                setTrainingInProgress(true);
            }
        } catch (e) {
            console.warn("System status check failed, retrying...");
        }
        return false;
    };

    // --- System Readiness Gate ---
    // This effect fires as soon as a user logs in.
    // It polls /health/status until all systems are ready, then hides the gate.
    useEffect(() => {
        if (!currentUser) return; // Only run when logged in

        // Show the gate while we check
        setGateVisible(true);
        setSystemReady(false);

        let pollInterval;

        const checkAndSchedule = async () => {
            const isReady = await checkSystemStatus();
            if (isReady) {
                clearInterval(pollInterval);
            } else {
                // pollInterval already running, just keep going
            }
        };

        checkAndSchedule(); // immediate first check
        pollInterval = setInterval(checkAndSchedule, 3000);

        return () => clearInterval(pollInterval);
    }, [currentUser]); // Only re-run on login/logout

    // Analytics fetch on tab switch
    useEffect(() => {
        if (view === 'analytics') {
            fetchAnalytics();
        }
    }, [view]);

    useEffect(() => {
        const fetchInitialState = async () => {
            if (!activeConversationId) return;
            try {
                const res = await axios.get(`${API_BASE}/conversation/${activeConversationId}/messages?limit=50`, {
                    headers: { 'X-API-Key': API_KEY }
                });
                if (res.data && res.data.length > 0) {
                    const historyMsgs = res.data.slice().reverse().map(m => {
                        const parsed = parseAnalysis(m);
                        const isSelf = currentUser && (m.sender_id === currentUser.user_id);
                        return {
                            id: m.id,
                            sender: isSelf ? 'user' : 'ai',
                            text: m.content,
                            senderName: isSelf ? currentUser.username : (m.sender_id ? m.sender_id.substring(0,8) : 'System'),
                            analysis: parsed
                        };
                    });
                    setMessages(historyMsgs);
                    const lastMsg = res.data[0];
                    const initialAnalysis = parseAnalysis(lastMsg);
                    if (initialAnalysis) {
                        setCurrentAnalysis(initialAnalysis);
                        applyTheme(initialAnalysis.data.final_dominant_emotion);
                    }
                    await fetchVibe();
                }
            } catch (err) { console.error("Initial fetch failed:", err); }
        };

        // check if ready
        let pollInterval;
        const connectWebSocket = () => {
            if (!currentUser) return null;
            const ws = new WebSocket(`${WS_BASE}/ws/${currentUser.user_id}?api_key=${API_KEY}`);

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

                        if (payload.vibe) {
                            setVibeAnalysis(payload.vibe);
                        } else {
                            fetchVibe();
                        }
                        
                        setMessages(prev => {
                            const isSelf = payload.data.sender_id === currentUser?.user_id;
                            if (prev.some(m => m.id === payload.data.id)) {
                                return prev.map(m => m.id === payload.data.id ? { ...m, analysis: payload } : m);
                            }
                            
                            const existingIdx = prev.findIndex(m => m.text === payload.data.raw_text && !m.analysis && isSelf && m.sender === 'user');
                            
                            const newMsgData = {
                                id: payload.data.id,
                                sender: isSelf ? 'user' : 'ai',
                                text: payload.data.raw_text,
                                senderName: isSelf ? currentUser?.username : (payload.data.sender_id ? payload.data.sender_id.substring(0,8) : 'System'),
                                analysis: payload
                            };

                            if (existingIdx >= 0) {
                                const newMsgs = [...prev];
                                newMsgs[existingIdx] = { ...newMsgs[existingIdx], ...newMsgData };
                                return newMsgs;
                            } else {
                                return [...prev, newMsgData];
                            }
                        });
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
            return ws;
        };

        const init = async () => {
            if (!currentUser || !systemReady) return; // Wait for system to be ready first
            connectWebSocket();
            await fetchInitialState();
        };
        init();

        return () => {
            if (socketRef.current) socketRef.current.close();
        };
    }, [activeConversationId, currentUser, systemReady]); // Re-run when convo changes, login, or system becomes ready

// Auth & Chat Fetching
useEffect(() => {
    if (!currentUser) return;
    const fetchChats = async () => {
        try {
            const headers = { 'X-API-Key': API_KEY };
            const [chatsRes, usersRes] = await Promise.all([
                axios.get(`${API_BASE}/conversations/${currentUser.user_id}`, { headers }),
                axios.get(`${API_BASE}/users?current_user_id=${currentUser.user_id}`, { headers })
            ]);
            setConversations(chatsRes.data || []);
            setGlobalUsers(usersRes.data || []);
        } catch (e) { console.error("Failed to load initial panel data"); }
    };
    fetchChats();
    const intl = setInterval(fetchChats, 5000);
    return () => clearInterval(intl);
}, [currentUser]);

const handleLogin = async (e) => {
    e.preventDefault();
    if (!loginUsername.trim()) return;
    try {
        const res = await axios.post(`${API_BASE}/login`, { username: loginUsername.trim() });
        setCurrentUser(res.data);
    } catch (err) { console.error("Login Error", err); }
};

const handleCreateChat = async (targetId) => {
    try {
        const res = await axios.post(`${API_BASE}/conversations`, {
            user_id: currentUser.user_id,
            target_user_id: targetId
        });
        setActiveConversationId(res.data.conversation_id);
        setMessages([]); // clear old messages
        setCurrentAnalysis(null);
        setActiveTab('chats');
    } catch (e) { console.error("Failed to make chat", e); }
};

const sendMessage = () => {
    if (!inputValue.trim()) return;

    // optimistic update
    const newMsg = {
        id: Date.now(),
        sender: 'user',
        text: inputValue,
        senderName: currentUser.username,
        analysis: null
    };
    setMessages(prev => [...prev, newMsg]);

    // send message over socket
    if (socketRef.current && activeConversationId) {
        socketRef.current.send(JSON.stringify({
            text: inputValue,
            recipient_id: 'system',
            sender_id: currentUser.user_id,
            conversation_id: activeConversationId
        }));
    }

    setInputValue('');
    setView('live');
};

// scroll on new message
useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
}, [messages, view]);


const handleFeedback = async (msgId, newLabel) => {
    try {
        await axios.post(`${API_BASE}/message/${msgId}/feedback`,
            { label: newLabel },
            { headers: { 'X-API-Key': API_KEY } }
        );
        // update state
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
        setView('live'); // show analysis
    } else {
        console.log("No analysis data for this message yet", msg);
    }
};


return (
    <div className="app-shell">
        {/* Login gate */}
        {!currentUser && (
            <div className="login-gate" style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.9)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div className="glass" style={{ padding: '40px', borderRadius: '20px', textAlign: 'center' }}>
                    <h2>Welcome to InnerLink</h2>
                    <form onSubmit={handleLogin}>
                        <input autoFocus placeholder="Enter Username" value={loginUsername} onChange={e => setLoginUsername(e.target.value)} style={{ padding: '10px', fontSize: '1.2rem', marginBottom: '20px', borderRadius: '5px', background: 'rgba(255,255,255,0.1)', color: 'white', border: '1px solid var(--accent-primary)' }} />
                        <br />
                        <button className="primary-btn" type="submit" style={{ padding: '10px 30px' }}>Login</button>
                    </form>
                </div>
            </div>
        )}

        {/* loading gate */}
        {(currentUser && !systemReady) && (
            <div className={`loading-gate ${gateVisible ? '' : 'gate-exit'}`}>
                <div className="gate-backdrop"></div>
                <div className="gate-content">
                    <div className="brain-loader">
                        <div className="pulse-ring"></div>
                        <div className="pulse-ring delay"></div>
                        <div className="brain-icon">🧠</div>
                    </div>
                    <h2 className="gate-title">InnerLink is Initializing</h2>
                    <p className="gate-subtitle">Warming up neural pipelines and calibrating the meta-learner...</p>

                    <div className="gate-checklist">
                        <div className={`gate-check-item ${componentStatus.redis ? 'ready' : 'pending'}`}>
                            <span className="check-icon">{componentStatus.redis ? '✓' : '◌'}</span>
                            <span className="check-label">Redis Stream Engine</span>
                        </div>
                        <div className={`gate-check-item ${componentStatus.database ? 'ready' : 'pending'}`}>
                            <span className="check-icon">{componentStatus.database ? '✓' : '◌'}</span>
                            <span className="check-label">PostgreSQL Database</span>
                        </div>
                        <div className={`gate-check-item ${componentStatus.meta_learner ? 'ready' : 'pending'}`}>
                            <span className="check-icon">{componentStatus.meta_learner ? '✓' : '◌'}</span>
                            <span className="check-label">Meta-Learner Intelligence</span>
                        </div>
                    </div>

                    <div className="gate-footer">
                        <div className="gate-spinner"></div>
                        <span>Checking every 3 seconds...</span>
                    </div>
                </div>
            </div>
        )}

        {/* Training in progress banner */}
        {systemReady && trainingInProgress && (
            <div style={{
                position: 'fixed', top: 0, left: 0, right: 0, zIndex: 500,
                background: 'linear-gradient(90deg, #b45309, #d97706)',
                color: '#fff', padding: '8px 20px',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                fontSize: '0.85rem', fontWeight: 500, boxShadow: '0 2px 8px rgba(0,0,0,0.4)'
            }}>
                <span>
                    🧠 <strong>AI is still learning</strong> — The meta-learner is training on new data.
                    Emotion analysis is live but accuracy will improve once training completes.
                </span>
                <button onClick={() => setTrainingInProgress(false)} style={{
                    background: 'rgba(255,255,255,0.2)', border: 'none', color: '#fff',
                    borderRadius: '4px', padding: '2px 10px', cursor: 'pointer', fontSize: '0.85rem'
                }}>✕ Dismiss</button>
            </div>
        )}

        {/* background */}
        <div className="bg-gradient"></div>
        <div className="glow-orb" id="orb-1"></div>
        <div className="glow-orb" id="orb-2"></div>

        {/* sidebar */}
        <aside className="sidebar glass">
            <header className="app-header">
                <div className="logo">
                    <div className="logo-icon"></div>
                    <h1>InnerLink</h1>
                </div>
                <div className="user-selector" style={{ display: 'none' }}>
                    {/* Hidden, replaced by modern tabs below */}
                </div>
                <div className="status-indicator">
                    <span className="pulse"></span> {status}
                </div>
            </header>

            {/* Custom User Info & Tabs */}
            {currentUser && (
                <div style={{ padding: '0 20px', marginBottom: '10px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <h3 style={{ color: 'var(--accent-primary)', marginBottom: '10px' }}>{currentUser.username}</h3>
                    </div>
                    <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                        <button onClick={() => setActiveTab('chats')} style={{ flex: 1, padding: '5px', background: activeTab === 'chats' ? 'rgba(255,255,255,0.2)' : 'transparent', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: '5px' }}>Chats</button>
                        <button onClick={() => setActiveTab('users')} style={{ flex: 1, padding: '5px', background: activeTab === 'users' ? 'rgba(255,255,255,0.2)' : 'transparent', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: '5px' }}>Users</button>
                    </div>

                    {activeTab === 'users' && (
                        <div className="users-list" style={{ maxHeight: '120px', overflowY: 'auto', marginBottom: '10px', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '5px' }}>
                            {globalUsers.map(u => (
                                <div key={u.user_id} onClick={() => handleCreateChat(u.user_id)} style={{ padding: '8px', cursor: 'pointer', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                    {u.username} <span style={{ float: 'right' }}>+</span>
                                </div>
                            ))}
                        </div>
                    )}
                    {activeTab === 'chats' && (
                        <div className="chats-list" style={{ maxHeight: '120px', overflowY: 'auto', marginBottom: '10px', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '5px' }}>
                            {conversations.map(c => (
                                <div key={c.conversation_id} onClick={() => setActiveConversationId(c.conversation_id)} style={{ padding: '8px', cursor: 'pointer', borderBottom: '1px solid rgba(255,255,255,0.05)', background: activeConversationId === c.conversation_id ? 'rgba(0,180,255,0.3)' : 'transparent' }}>
                                    <div style={{ fontWeight: 'bold' }}>Chat with {c.other_username}</div>
                                    <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '4px' }}>
                                        Vibe: <span style={{ color: EMOTION_COLORS[c.dominant_emotion?.toLowerCase()] || '#00f2ff' }}>{c.dominant_emotion || 'Neutral'}</span> (Valence: {(c.average_valence || 0).toFixed(2)})
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

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
                        <div>{msg.text}</div>
                        {msg.analysis && (
                            <div style={{ fontSize: '0.75rem', marginTop: '6px', opacity: 0.9 }}>
                                <span style={{ color: EMOTION_COLORS[msg.analysis.data.final_dominant_emotion?.toLowerCase()] || '#00f2ff', fontWeight: 'bold' }}>
                                    {msg.analysis.data.final_dominant_emotion}
                                </span>
                                {msg.analysis.data.bert_emotions && msg.analysis.data.bert_emotions.length > 0 && (
                                    <span style={{ marginLeft: '6px', color: '#94a3b8' }}>
                                        ({msg.analysis.data.bert_emotions.slice(0, 2).map(e => e.label).join(', ')})
                                    </span>
                                )}
                            </div>
                        )}
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

        {/* main dashboard */}
        <main className="dashboard">
            <div className="dashboard-header">
                <h2>Discovery Module</h2>
                <div className="analysis-meta">
                    {currentAnalysis ? `ID: ${currentAnalysis.data.id.split('-')[0]}` : 'Ready'}
                </div>
            </div>

            {/* idle state */}
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

            {/* active analysis */}
            {currentAnalysis && view === 'live' && (
                <div className="analysis-view">
                    {/* vibe section */}
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

                    {/* dominant emotion */}
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

                        {/* word highlights */}
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
                        {/* bert breakdown */}
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

                        {/* conflict / sarcasm meter */}
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

                        {/* text insight */}
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
                            {Object.entries(analyticsData?.emotion_breakdown || {}).sort((a, b) => b[1].samples - a[1].samples).map(([emo, stats]) => (
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
