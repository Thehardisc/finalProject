import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import LoginModal from './components/LoginModal';
import DemoRunner from './components/DemoRunner';
import IGDashboard from './pages/IGDashboard';
import AnalyticsPage from './pages/AnalyticsPage';
import LiveAnalyticsDashboardPage from './pages/LiveAnalyticsDashboardPage';
import AdminPipelinePage from './pages/AdminPipelinePage';
import LandingPage from './pages/LandingPage';
import './glass/CrystalGlass-v2.css';

const API_BASE = import.meta.env.VITE_API_URL  || 'http://localhost:8001';
const WS_BASE  = import.meta.env.VITE_WS_URL   || 'ws://localhost:8001';

axios.defaults.withCredentials = true;

// ── Covert Micro-Delay (friction) ─────────────────────────────────────────────
const FRICTION_DELAY_MS = 280;
const FRICTION_CDM_STATES = new Set(['TENSION', 'CONFLICT', 'ARGUMENT', 'FRUSTRATION']);

function shouldApplyFriction(data) {
  if (!data) return false;
  const cdmState = data.context_snapshot?.cdm_current_state;
  const sarcasm  = data.sarcasm_score  ?? 0;
  const inertia  = data.dynamics?.inertia ?? 0;
  const valence  = data.vad?.valence      ?? 0;
  return (sarcasm > 0.75 && FRICTION_CDM_STATES.has(cdmState))
      || (inertia > 0.80 && valence < -0.5);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseAnalysis(msg) {
  if (!msg.emotions) return null;
  try {
    const ems = typeof msg.emotions === 'string' ? JSON.parse(msg.emotions) : msg.emotions;
    const pl  = msg.pipeline_log
      ? (typeof msg.pipeline_log === 'string' ? JSON.parse(msg.pipeline_log) : msg.pipeline_log)
      : {};
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
        id:                     msg.id,
        raw_text:               msg.content || msg.text,
        final_dominant_emotion: ems.dominant_emotion || pl.dominant_selected || 'Neutral',
        final_valence:          pl.models?.vader?.vader_compound ?? pl.vad?.valence ?? 0,
        bert_emotions:          bert_list,
        llm_insights:           'Analysis loaded.',
        sarcasm_score:          pl.sarcasm_score || 0,
        inversion_applied:      pl.inversion_applied || false,
        hierarchical_scores:    [],
        emojis_found:           [],
        slang_detected:         {},
        meta_confidence:        pl.meta_confidence ?? null,
        logic_map:              pl.logic_map || null,
        gate_weights_alpha:     pl.gate_weights_alpha || null,
        ekman_group:            pl.ekman_group || null,
        context_snapshot:       pl.context_snapshot || null,
        context_shift:          msg.context_shift
                                  ? (typeof msg.context_shift === 'string' ? JSON.parse(msg.context_shift) : msg.context_shift)
                                  : null,
        lstm_trajectory:        pl.trajectory || null,
        vad:                    pl.vad || {},
        dynamics:               pl.dynamics || {},
        appraisal:              pl.appraisal || {},
      },
    };
  } catch {
    return null;
  }
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [view, setView] = useState('dashboard'); // 'dashboard' | 'analytics' | 'live-analytics' | 'admin'
  const [showLoginModal, setShowLoginModal] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.body.style.background = '#05060F';
  }, []);

  // ── Auth ────────────────────────────────────────────────────────────────
  const [currentUser, setCurrentUser]       = useState(null);
  const [sessionChecked, setSessionChecked] = useState(false);

  // ── System readiness ────────────────────────────────────────────────────
  const [systemReady, setSystemReady]               = useState(false);
  const [gateVisible, setGateVisible]               = useState(false);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [componentStatus, setComponentStatus]       = useState({
    database: false, redis: false, meta_learner: false,
  });

  // ── Data ────────────────────────────────────────────────────────────────
  const [conversations, setConversations]           = useState([]);
  const [globalUsers, setGlobalUsers]               = useState([]);
  const [onlineUsers, setOnlineUsers]               = useState(new Set());
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages]                     = useState([]);
  const [inputValue, setInputValue]                 = useState('');
  const [status, setStatus]                         = useState('Offline');
  const [currentAnalysis, setCurrentAnalysis]       = useState(null);
  const [analyticsData, setAnalyticsData]           = useState(null);
  const [mlProcessing, setMlProcessing]             = useState(false);
  const [regeneratingIds, setRegeneratingIds]       = useState(new Set());
  const [partialModels, setPartialModels]           = useState(new Set());

  const socketRef      = useRef(null);
  const activeConvRef  = useRef(activeConversationId); // tracks current conv without causing WS reconnect
  const retryCountRef  = useRef(0);
  const isSendingRef   = useRef(false);
  const retryTimerRef  = useRef(null);
  const mountedRef     = useRef(true);

  // Keep ref in sync whenever the active conversation changes
  useEffect(() => {
    activeConvRef.current = activeConversationId;
  }, [activeConversationId]);

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
    mountedRef.current = false;
    clearTimeout(retryTimerRef.current);
    if (socketRef.current) { socketRef.current.close(1000, 'logout'); socketRef.current = null; }
    setCurrentUser(null);
    setMessages([]);
    setConversations([]);
    setActiveConversationId(null);
    setCurrentAnalysis(null);
    setSystemReady(false);
    setView('dashboard');
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
  const fetchConversations = async () => {
    if (!currentUser) return;
    try {
      const [chats, users, online] = await Promise.all([
        axios.get(`${API_BASE}/conversations/${currentUser.user_id}`),
        axios.get(`${API_BASE}/users?current_user_id=${currentUser.user_id}`),
        axios.get(`${API_BASE}/users/online`),
      ]);
      // Filter out any self-conversations that may exist from legacy data
      const all = chats.data || [];
      setConversations(all.filter(c => c.type === 'group' || c.other_user_id !== currentUser.user_id));
      setGlobalUsers(users.data || []);
      setOnlineUsers(new Set(online.data?.online_user_ids || []));
    } catch {}
  };

  useEffect(() => {
    if (!currentUser) return;
    fetchConversations();
    const id = setInterval(fetchConversations, 5000);
    return () => clearInterval(id);
  }, [currentUser]);

  // ── Persistent WebSocket — one connection per session, not per conversation ──
  useEffect(() => {
    if (!currentUser || !systemReady) return;

    mountedRef.current    = true;
    retryCountRef.current = 0;

    const MAX_RETRIES = 8;
    const BASE_DELAY  = 1000;

    function connect() {
      if (!mountedRef.current) return;

      const ws = new WebSocket(`${WS_BASE}/ws/${currentUser.user_id}`);
      socketRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) { ws.close(); return; }
        retryCountRef.current = 0;
        setStatus('Live');
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const payload = JSON.parse(event.data);

          if (payload.type === 'analysis') {
            // Only update UI if this message belongs to the active conversation
            const convId = payload.data?.conversation_id;
            if (convId && convId !== activeConvRef.current) return;

            setMlProcessing(false);
            setPartialModels(new Set());
            setRegeneratingIds(prev => { const n = new Set(prev); n.delete(payload.data?.id); return n; });
            setCurrentAnalysis(prev => ({ ...payload, ai_insight: null, loadingReasoning: true }));

            setMessages(prev => {
              const isSelf = payload.data.sender_id === currentUser.user_id;
              const newMsg = {
                id:         payload.data.id,
                sender:     isSelf ? 'user' : 'ai',
                text:       payload.data.raw_text,
                senderName: isSelf ? currentUser.display_name : payload.data.sender_id?.substring(0, 8),
                analysis:   payload,
              };

              if (prev.some(m => m.id === payload.data.id)) {
                return prev.map(m => m.id === payload.data.id ? { ...m, analysis: payload } : m);
              }

              const optimisticIdx = isSelf
                ? prev.findIndex(m => m.text === payload.data.raw_text && !m.analysis && m.sender === 'user')
                : -1;

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
            setMessages(prev => prev.map(m =>
              m.id === payload.message_id
                ? { ...m, analysis: { ...m.analysis, ai_insight: payload.ai_insight } }
                : m
            ));
          } else if (payload.type === 'model_ready') {
            setPartialModels(prev => {
              const next = new Set(prev);
              next.add(payload.model);
              return next;
            });
          }
        } catch {}
      };

      ws.onclose = (event) => {
        if (!mountedRef.current) return;
        setStatus('Offline');
        if (event.code === 1000) return;
        if (event.code === 1008) { handleLogout(); return; }
        if (retryCountRef.current < MAX_RETRIES) {
          const delay = Math.min(BASE_DELAY * 2 ** retryCountRef.current, 30_000);
          retryCountRef.current++;
          setStatus(`Reconnecting (${retryCountRef.current}/${MAX_RETRIES})…`);
          retryTimerRef.current = setTimeout(connect, delay);
        } else {
          setStatus('Connection failed — please refresh');
        }
      };

      ws.onerror = () => ws.close();
    }

    connect();

    return () => {
      mountedRef.current = false;
      clearTimeout(retryTimerRef.current);
      if (socketRef.current) socketRef.current.close(1000, 'component unmounted');
    };
  }, [currentUser, systemReady]); // ← activeConversationId intentionally excluded

  // ── History load on conversation switch ─────────────────────────────────
  useEffect(() => {
    if (!currentUser || !systemReady || !activeConversationId) return;

    const loadHistory = async () => {
      try {
        const res = await axios.get(`${API_BASE}/conversation/${activeConversationId}/messages?limit=50`);
        if (res.data?.length > 0) {
          const msgs = res.data.slice().reverse().map(m => {
            const isSelf = m.sender_id === currentUser.user_id;
            return {
              id:         m.id,
              sender:     isSelf ? 'user' : 'ai',
              text:       m.content,
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
        } else {
          setMessages([]);
          setCurrentAnalysis(null);
        }
      } catch {}
    };

    loadHistory();
  }, [activeConversationId, currentUser, systemReady]);

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
        user_id:        currentUser.user_id,
        target_user_id: targetId,
      });
      await fetchConversations();
      setActiveConversationId(res.data.conversation_id);
      setMessages([]);
      setCurrentAnalysis(null);
      setView('dashboard');
    } catch {}
  };

  const handleCreateGroup = async (name, memberIds) => {
    const res = await axios.post(`${API_BASE}/conversations/group`, {
      name,
      member_ids: memberIds,
    });
    await fetchConversations();
    setActiveConversationId(res.data.conversation_id);
    setMessages([]);
    setCurrentAnalysis(null);
    setView('dashboard');
  };

  const handleAddMember = async (convId, userId) => {
    await axios.post(`${API_BASE}/conversations/${convId}/members`, { user_id: userId });
    await fetchConversations();
  };

  const handleRemoveMember = async (convId, userId) => {
    await axios.delete(`${API_BASE}/conversations/${convId}/members/${userId}`);
    await fetchConversations();
  };

  const handleSelectConversation = (convId) => {
    setActiveConversationId(convId);
    setMessages([]);
    setCurrentAnalysis(null);
    setView('dashboard');
  };

  const handleDeleteMessage = async (msgId) => {
    try {
      await axios.delete(`${API_BASE}/message/${msgId}`);
      setMessages(prev => prev.filter(m => m.id !== msgId));
      setCurrentAnalysis(prev => prev?.data?.id === msgId ? null : prev);
    } catch {}
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || !socketRef.current || !activeConversationId) return;
    if (isSendingRef.current) return;
    isSendingRef.current = true;

    const text = inputValue;

    try {
      if (shouldApplyFriction(currentAnalysis?.data)) {
        await new Promise(r => setTimeout(r, FRICTION_DELAY_MS));
      }

      const optimistic = {
        id:         Date.now(),
        sender:     'user',
        text,
        senderName: currentUser.display_name,
        analysis:   null,
      };
      setMessages(prev => [...prev, optimistic]);
      setMlProcessing(true);
      setPartialModels(new Set());
      socketRef.current.send(JSON.stringify({
        text,
        recipient_id:    'system',
        sender_id:       currentUser.user_id,
        conversation_id: activeConversationId,
      }));
      setInputValue('');
    } finally {
      isSendingRef.current = false;
    }
  };

  const handleRegenerateAnalysis = (msgId) => {
    setRegeneratingIds(prev => new Set([...prev, msgId]));
    setMlProcessing(true);
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

  const handleDemoStart = async (convId) => {
    await fetchConversations();
    setActiveConversationId(convId);
    setMessages([]);
    setCurrentAnalysis(null);
  };

  // ── Render: loading gate ──────────────────────────────────────────────────
  if (currentUser && !systemReady) {
    const ready = Object.values(componentStatus).filter(Boolean).length;
    const total = Math.max(Object.keys(componentStatus).length, 1);
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: '#05060F',
        fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      }}>
        <div style={{
          position: 'fixed', width: 700, height: 700, borderRadius: '50%',
          left: -250, top: -250,
          background: 'radial-gradient(circle, rgba(79,40,210,0.38) 0%, transparent 70%)',
          filter: 'blur(100px)', pointerEvents: 'none',
        }} />
        <div style={{
          position: 'fixed', width: 600, height: 600, borderRadius: '50%',
          right: -200, bottom: -200,
          background: 'radial-gradient(circle, rgba(46,16,130,0.48) 0%, transparent 70%)',
          filter: 'blur(90px)', pointerEvents: 'none',
        }} />
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          textAlign: 'center', position: 'relative', zIndex: 1,
        }}>
          <div style={{
            width: 52, height: 52, borderRadius: 18, marginBottom: 28,
            background: 'linear-gradient(135deg, #7c3aed 0%, #4338ca 100%)',
            boxShadow: '0 4px 24px rgba(109,40,217,0.50)',
          }} />
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: 'rgba(255,255,255,0.88)', letterSpacing: '-0.02em', marginBottom: 6 }}>
            InnerLink
          </div>
          <div style={{ fontSize: '0.82rem', color: 'rgba(255,255,255,0.38)', marginBottom: 40 }}>
            Preparing your secure session
          </div>
          <div style={{ position: 'relative', width: 56, height: 56, marginBottom: 28 }}>
            <svg width="56" height="56" style={{ transform: 'rotate(-90deg)' }}>
              <circle cx="28" cy="28" r="22" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="3" />
              <circle
                cx="28" cy="28" r="22" fill="none" stroke="url(#gate-grad)"
                strokeWidth="3" strokeLinecap="round"
                strokeDasharray={`${2 * Math.PI * 22}`}
                strokeDashoffset={`${2 * Math.PI * 22 * (1 - ready / total)}`}
                style={{ transition: 'stroke-dashoffset 0.6s ease' }}
              />
              <defs>
                <linearGradient id="gate-grad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#6d28d9" />
                  <stop offset="100%" stopColor="#4c1d95" />
                </linearGradient>
              </defs>
            </svg>
            <div style={{
              position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
              justifyContent: 'center', fontSize: '0.72rem', fontWeight: 700,
              color: 'rgba(167,139,250,0.88)',
            }}>
              {ready}/{total}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {Object.entries(componentStatus).map(([key, done]) => (
              <div key={key} style={{
                width: 8, height: 8, borderRadius: '50%',
                background: done ? '#4ade80' : 'rgba(255,255,255,0.15)',
                boxShadow: done ? '0 0 8px rgba(74,222,128,0.50)' : 'none',
                transition: 'all 0.4s ease',
              }} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  // ── Render: login / landing ───────────────────────────────────────────────
  if (!sessionChecked) return null;
  if (!currentUser) return (
    <>
      <LandingPage onSignIn={() => setShowLoginModal(true)} />
      {showLoginModal && (
        <LoginModal onSuccess={handleAuthSuccess} onClose={() => setShowLoginModal(false)} />
      )}
    </>
  );

  // ── Render: main app ──────────────────────────────────────────────────────
  return (
    <div className="crystal-shell">
      {trainingInProgress && (
        <div className="training-banner">
          <span>
            <strong>AI is still learning</strong> — Meta-learner is training.
            Emotion analysis is live; accuracy improves once training completes.
          </span>
          <button onClick={() => setTrainingInProgress(false)}>✕ Dismiss</button>
        </div>
      )}

      {view === 'dashboard' && (
        <IGDashboard
          currentUser={currentUser}
          conversations={conversations}
          globalUsers={globalUsers}
          onlineUsers={onlineUsers}
          activeConversationId={activeConversationId}
          onSelectConversation={handleSelectConversation}
          onCreateChat={handleCreateChat}
          onCreateGroup={handleCreateGroup}
          onAddMember={handleAddMember}
          onRemoveMember={handleRemoveMember}
          onDeleteMessage={handleDeleteMessage}
          onGoToAnalytics={() => setView('analytics')}
          onGoToLiveAnalytics={() => setView('live-analytics')}
          onGoToAdmin={() => setView('admin')}
          onLogout={handleLogout}
          status={status}
          messages={messages}
          inputValue={inputValue}
          setInputValue={setInputValue}
          onSend={sendMessage}
          currentAnalysis={currentAnalysis}
          processing={mlProcessing}
          partialModels={partialModels}
          regeneratingIds={regeneratingIds}
          onRegenerateAnalysis={handleRegenerateAnalysis}
          onInjectDemo={handleInjectDemo}
          socketRef={socketRef}
          onDemoStart={handleDemoStart}
        />
      )}

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

      {view === 'live-analytics' && (
        <LiveAnalyticsDashboardPage
          currentUser={currentUser}
          onBack={() => setView('dashboard')}
        />
      )}

      {view === 'admin' && (
        <AdminPipelinePage
          currentUser={currentUser}
          onBack={() => setView('dashboard')}
        />
      )}
    </div>
  );
}
