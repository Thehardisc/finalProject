import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import LoginModal from './components/LoginModal';
import IGDashboard from './pages/IGDashboard';
import AnalyticsPage from './pages/AnalyticsPage';
import './glass/CrystalGlass.css';

const API_BASE = 'http://localhost:8001';
const WS_BASE  = 'ws://localhost:8001';

axios.defaults.withCredentials = true;

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseAnalysis(msg) {
  if (!msg.emotions) return null;
  try {
    const ems = typeof msg.emotions === 'string' ? JSON.parse(msg.emotions) : msg.emotions;
    const bert_list = [];
    for (const [k, v] of Object.entries(ems)) {
      if (!['vader_neg','vader_neu','vader_pos','vader_compound',
            'dominant_emotion','sentiment_positive','sentiment_negative'].includes(k)) {
        bert_list.push({ label: k, score: v });
      }
    }
    return {
      type: 'analysis',
      data: {
        id: msg.id,
        raw_text: msg.content || msg.text,
        final_dominant_emotion: ems.dominant_emotion || 'Neutral',
        final_valence: ems.vader_compound || 0,
        bert_emotions: bert_list,
        llm_insights: 'Analysis loaded.',
        llm_sarcasm_score: 0,
        hierarchical_scores: [],
        emojis_found: [],
        slang_detected: {},
      },
    };
  } catch {
    return null;
  }
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  // ── Navigation ──────────────────────────────────────────────────────────
  const [view, setView] = useState('dashboard'); // 'dashboard' | 'analytics'

  // ── Auth ────────────────────────────────────────────────────────────────
  const [currentUser, setCurrentUser]       = useState(null);
  const [sessionChecked, setSessionChecked] = useState(false);

  // ── System readiness ────────────────────────────────────────────────────
  const [systemReady, setSystemReady]         = useState(false);
  const [gateVisible, setGateVisible]         = useState(false);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [componentStatus, setComponentStatus] = useState({
    database: false, redis: false, meta_learner: false,
  });

  // ── Data ────────────────────────────────────────────────────────────────
  const [conversations, setConversations]       = useState([]);
  const [globalUsers, setGlobalUsers]           = useState([]);
  const [onlineUsers, setOnlineUsers]           = useState(new Set());
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages]                 = useState([]);
  const [inputValue, setInputValue]             = useState('');
  const [status, setStatus]                     = useState('Offline');
  const [currentAnalysis, setCurrentAnalysis]   = useState(null);
  const [analyticsData, setAnalyticsData]       = useState(null);
  const [mlProcessing, setMlProcessing]         = useState(false);
  const [regeneratingIds, setRegeneratingIds]   = useState(new Set());

  const socketRef = useRef(null);

  // ── Session restore ──────────────────────────────────────────────────────
  useEffect(() => {
    if (currentUser) return;
    axios.get(`${API_BASE}/auth/me`)
      .then(res => setCurrentUser({
        user_id:      res.data.user_id,
        display_name: res.data.display_name,
        email:        res.data.email,
        role:         res.data.role,
      }))
      .catch(() => {})
      .finally(() => setSessionChecked(true));
  }, [currentUser]);

  // ── Auth helpers ─────────────────────────────────────────────────────────
  const handleAuthSuccess = (data) => {
    setCurrentUser({
      user_id:      data.user_id,
      display_name: data.display_name,
      email:        data.email,
      role:         data.role,
    });
  };

  const handleLogout = async () => {
    try { await axios.post(`${API_BASE}/auth/logout`); } catch {}
    setCurrentUser(null);
    setMessages([]);
    setConversations([]);
    setActiveConversationId(null);
    setCurrentAnalysis(null);
    setSystemReady(false);
    setView('dashboard');
    if (socketRef.current) { socketRef.current.close(); socketRef.current = null; }
  };

  // ── System readiness gate ────────────────────────────────────────────────
  useEffect(() => {
    if (!currentUser) return;
    setGateVisible(true);
    setSystemReady(false);

    const checkStatus = async () => {
      try {
        const res = await axios.get(`${API_BASE}/health/status`, {
          validateStatus: s => s === 200 || s === 503,
        });
        const data = res.data;
        if (data.components) setComponentStatus(data.components);
        setTrainingInProgress(data.training_in_progress === true);
        if (data.ready) {
          setGateVisible(false);
          setTimeout(() => setSystemReady(true), 600);
          return true;
        }
      } catch {}
      return false;
    };

    let interval;
    checkStatus().then(ready => {
      if (!ready) interval = setInterval(async () => {
        const done = await checkStatus();
        if (done) clearInterval(interval);
      }, 3000);
    });
    return () => clearInterval(interval);
  }, [currentUser]);

  // ── Conversations + Users polling ───────────────────────────────────────
  useEffect(() => {
    if (!currentUser) return;
    const fetchAll = async () => {
      try {
        const [chats, users, online] = await Promise.all([
          axios.get(`${API_BASE}/conversations/${currentUser.user_id}`),
          axios.get(`${API_BASE}/users?current_user_id=${currentUser.user_id}`),
          axios.get(`${API_BASE}/users/online`),
        ]);
        setConversations(chats.data || []);
        setGlobalUsers(users.data || []);
        setOnlineUsers(new Set(online.data?.online_user_ids || []));
      } catch {}
    };
    fetchAll();
    const id = setInterval(fetchAll, 5000);
    return () => clearInterval(id);
  }, [currentUser]);

  // ── WebSocket + conversation history ────────────────────────────────────
  useEffect(() => {
    if (!currentUser || !systemReady || !activeConversationId) return;

    // Load history
    const loadHistory = async () => {
      try {
        const res = await axios.get(`${API_BASE}/conversation/${activeConversationId}/messages?limit=50`);
        if (res.data?.length > 0) {
          const msgs = res.data.slice().reverse().map(m => {
            const isSelf = m.sender_id === currentUser.user_id;
            return {
              id: m.id,
              sender: isSelf ? 'user' : 'ai',
              text: m.content,
              senderName: isSelf ? currentUser.display_name : (
                conversations.find(c => c.other_user_id === m.sender_id)?.other_display_name
                || m.sender_id.substring(0, 8)
              ),
              analysis: parseAnalysis(m),
            };
          });
          setMessages(msgs);
          const last = parseAnalysis(res.data[0]);
          if (last) setCurrentAnalysis(last);
        }
      } catch {}
    };

    // Open WS
    if (socketRef.current) socketRef.current.close();
    const ws = new WebSocket(`${WS_BASE}/ws/${currentUser.user_id}`);

    ws.onopen  = () => setStatus('Live');
    ws.onclose = () => setStatus('Offline');

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);

        if (payload.type === 'analysis') {
          setMlProcessing(false);
          setRegeneratingIds(prev => { const n = new Set(prev); n.delete(payload.data?.id); return n; });
          setCurrentAnalysis(prev => ({ ...payload, ai_insight: null, loadingReasoning: true }));

          setMessages(prev => {
            const isSelf = payload.data.sender_id === currentUser.user_id;
            if (prev.some(m => m.id === payload.data.id)) {
              return prev.map(m => m.id === payload.data.id ? { ...m, analysis: payload } : m);
            }
            const optimisticIdx = prev.findIndex(
              m => m.text === payload.data.raw_text && !m.analysis && isSelf && m.sender === 'user'
            );
            const newMsg = {
              id: payload.data.id,
              sender: isSelf ? 'user' : 'ai',
              text: payload.data.raw_text,
              senderName: isSelf
                ? currentUser.display_name
                : (conversations.find(c => c.other_user_id === payload.data.sender_id)?.other_display_name
                   || payload.data.sender_id?.substring(0, 8)),
              analysis: payload,
            };
            if (optimisticIdx >= 0) {
              const next = [...prev];
              next[optimisticIdx] = { ...next[optimisticIdx], ...newMsg };
              return next;
            }
            return [...prev, newMsg];
          });
        } else if (payload.type === 'reasoning') {
          setCurrentAnalysis(prev => {
            if (prev?.data?.id === payload.message_id) {
              return { ...prev, ai_insight: payload.ai_insight, loadingReasoning: false };
            }
            return prev;
          });
        }
      } catch {}
    };

    socketRef.current = ws;
    loadHistory();

    return () => { ws.close(); };
  }, [activeConversationId, currentUser, systemReady]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Analytics fetch ──────────────────────────────────────────────────────
  const fetchAnalytics = async () => {
    try {
      const res = await axios.get(`${API_BASE}/analytics/calibration`);
      setAnalyticsData(res.data);
    } catch {}
  };

  useEffect(() => {
    if (view === 'analytics') fetchAnalytics();
  }, [view]);

  // ── Chat actions ─────────────────────────────────────────────────────────
  const handleCreateChat = async (targetId) => {
    try {
      const res = await axios.post(`${API_BASE}/conversations`, {
        user_id: currentUser.user_id,
        target_user_id: targetId,
      });
      setActiveConversationId(res.data.conversation_id);
      setMessages([]);
      setCurrentAnalysis(null);
      setView('dashboard');
    } catch {}
  };

  const handleSelectConversation = (convId) => {
    setActiveConversationId(convId);
    setMessages([]);
    setCurrentAnalysis(null);
    setView('dashboard');
  };

  const sendMessage = () => {
    if (!inputValue.trim() || !socketRef.current || !activeConversationId) return;
    const optimistic = {
      id: Date.now(),
      sender: 'user',
      text: inputValue,
      senderName: currentUser.display_name,
      analysis: null,
    };
    setMessages(prev => [...prev, optimistic]);
    setMlProcessing(true);
    socketRef.current.send(JSON.stringify({
      text: inputValue,
      recipient_id: 'system',
      sender_id: currentUser.user_id,
      conversation_id: activeConversationId,
    }));
    setInputValue('');
  };

  const handleRegenerateAnalysis = (msgId) => {
    setRegeneratingIds(prev => new Set([...prev, msgId]));
    setMlProcessing(true);
    // Visual simulation — clears after pipeline animation completes (~2.8s)
    setTimeout(() => {
      setRegeneratingIds(prev => { const n = new Set(prev); n.delete(msgId); return n; });
      setMlProcessing(false);
    }, 2800);
  };

  const handleInjectDemo = (demoMessages) => {
    setMessages(demoMessages);
    const last = demoMessages[demoMessages.length - 1];
    if (last?.analysis) setCurrentAnalysis(last.analysis);
  };

  // ── Render: loading gate ──────────────────────────────────────────────────
  if (currentUser && !systemReady) {
    return (
      <div className="crystal-shell">
        <div className="crystal-gate">
          <div style={{ '--bubble-rgb': '0, 119, 255' }}>
            <div style={{
              width: 48, height: 48, borderRadius: 16,
              background: 'linear-gradient(135deg, #0077ff 0%, #7000ff 100%)',
              boxShadow: '0 6px 20px rgba(0,119,255,0.35)',
              marginBottom: 24,
            }} />
          </div>
          <h2 style={{ margin: '0 0 6px', fontSize: '1.2rem', fontWeight: 700, color: '#1c1c2e' }}>
            InnerLink is Initializing
          </h2>
          <p style={{ margin: '0 0 24px', color: '#6b7280', fontSize: '0.88rem' }}>
            Warming up neural pipelines…
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: 260 }}>
            {[
              { key: 'redis',        label: 'Redis Stream Engine' },
              { key: 'database',     label: 'PostgreSQL Database' },
              { key: 'meta_learner', label: 'Meta-Learner Intelligence' },
            ].map(({ key, label }) => (
              <div key={key} className={`crystal-gate-check ${componentStatus[key] ? 'ready' : ''}`}>
                <span style={{ fontSize: '0.85rem' }}>{componentStatus[key] ? '✓' : '◌'}</span>
                {label}
              </div>
            ))}
          </div>

          <div style={{ marginTop: 28, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
            <div className="crystal-gate-spinner" style={{ '--bubble-rgb': '0, 119, 255' }} />
            <span style={{ fontSize: '0.78rem', color: '#9ca3af' }}>Checking every 3 seconds…</span>
          </div>
        </div>
      </div>
    );
  }

  // ── Render: login ─────────────────────────────────────────────────────────
  if (!sessionChecked) return null;
  if (!currentUser) return <LoginModal onSuccess={handleAuthSuccess} />;

  // ── Render: main app ──────────────────────────────────────────────────────
  return (
    <div className="crystal-shell">
      {/* Training banner */}
      {trainingInProgress && (
        <div className="training-banner">
          <span>
            <strong>AI is still learning</strong> — Meta-learner is training.
            Emotion analysis is live; accuracy improves once training completes.
          </span>
          <button onClick={() => setTrainingInProgress(false)}>✕ Dismiss</button>
        </div>
      )}

      {/* Main — Instagram DM layout */}
      {view === 'dashboard' && (
        <IGDashboard
          currentUser={currentUser}
          conversations={conversations}
          globalUsers={globalUsers}
          onlineUsers={onlineUsers}
          activeConversationId={activeConversationId}
          onSelectConversation={handleSelectConversation}
          onCreateChat={handleCreateChat}
          onGoToAnalytics={() => setView('analytics')}
          onLogout={handleLogout}
          status={status}
          messages={messages}
          inputValue={inputValue}
          setInputValue={setInputValue}
          onSend={sendMessage}
          currentAnalysis={currentAnalysis}
          processing={mlProcessing}
          regeneratingIds={regeneratingIds}
          onRegenerateAnalysis={handleRegenerateAnalysis}
          onInjectDemo={handleInjectDemo}
        />
      )}

      {/* Analytics */}
      {view === 'analytics' && (
        <AnalyticsPage
          analyticsData={analyticsData}
          messages={messages}
          currentAnalysis={currentAnalysis}
          currentUser={currentUser}
          onBack={() => setView('dashboard')}
          onRefresh={fetchAnalytics}
        />
      )}
    </div>
  );
}
