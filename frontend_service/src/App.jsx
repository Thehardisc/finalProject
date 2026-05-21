import React, { useState, useEffect, useRef } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, Title, Filler } from 'chart.js';
import { authAPI, systemAPI, WS_BASE } from './api/client';
import LoginModal from './components/LoginModal';
import AdminDashboard from './components/AdminDashboard';
import GroupModal from './components/GroupModal';
import { EMOTION_COLORS } from './constants/emotions';
import { useSystemReadiness } from './hooks/useSystemReadiness';
import { useConversations }   from './hooks/useConversations';
import { useMessages }        from './hooks/useMessages';
import { useWebSocket }       from './hooks/useWebSocket';
import LiveView               from './views/LiveView';
import HistoryView            from './views/HistoryView';
import AnalyticsView          from './views/AnalyticsView';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, Title, Filler);


function App() {
    const [view,                setView]                = useState('live');
    const [status,              setStatus]              = useState('Offline');
    const [currentUser,         setCurrentUser]         = useState(null);
    const [sessionChecked,      setSessionChecked]      = useState(false);
    const [activeConversationId,setActiveConversationId]= useState(null);
    const [activeTab,           setActiveTab]           = useState('chats');
    const [showEmojiPicker,     setShowEmojiPicker]     = useState(false);
    const [analyticsData,       setAnalyticsData]       = useState(null);
    const [showGroupModal,      setShowGroupModal]      = useState(false);
    const [memberPanelOpen,     setMemberPanelOpen]     = useState(false);
    const [memberActionError,   setMemberActionError]   = useState('');

    const chatContainerRef = useRef(null);

    // ── Custom hooks ────────────────────────────────────────────────────────
    const { systemReady, gateVisible, trainingInProgress, componentStatus,
            setTrainingInProgress } = useSystemReadiness(currentUser);

    const {
        conversations, globalUsers,
        handleCreateChat, handleCreateGroup,
        handleAddMember, handleRemoveMember
    } = useConversations(currentUser);

    const {
        messages, setMessages, currentAnalysis, setCurrentAnalysis,
        vibeAnalysis, setVibeAnalysis, inputValue, setInputValue,
        socketRef, messagesEndRef, fetchVibe, getSenderName,
        sendMessage, handleFeedback, handleHistoryClick, handleDeleteMessage
    } = useMessages(currentUser, activeConversationId, systemReady, conversations, globalUsers);

    // ── Active conversation metadata ────────────────────────────────────────
    const activeConv = conversations.find(c => c.conversation_id === activeConversationId) || null;
    const isGroup    = activeConv?.type === 'group';
    const isCreator  = isGroup && activeConv?.creator_user_id === currentUser?.user_id;

    // ── Theme helper ─────────────────────────────────────────────────────────
    const applyTheme = (emotion) => {
        const color = EMOTION_COLORS[emotion?.toLowerCase()] || '#00f2ff';
        document.documentElement.style.setProperty('--accent-primary', color);
    };

    useWebSocket({
        currentUser, activeConversationId, systemReady,
        setStatus, setMessages, setCurrentAnalysis, setVibeAnalysis,
        applyTheme, fetchVibe, socketRef
    });

    // ── Session restore ──────────────────────────────────────────────────────
    useEffect(() => {
        const verifySession = async () => {
            if (!currentUser) {
                try {
                    const res = await authAPI.me();
                    setCurrentUser({
                        user_id:      res.data.user_id,
                        display_name: res.data.display_name,
                        email:        res.data.email,
                        role:         res.data.role
                    });
                } catch (e) { /* no active session */ }
                finally { setSessionChecked(true); }
            }
        };
        verifySession();
    }, [currentUser]);

    // ── Analytics fetch ──────────────────────────────────────────────────────
    useEffect(() => {
        if (view === 'analytics') {
            systemAPI.calibration()
                .then(r => setAnalyticsData(r.data))
                .catch(e => console.error('Failed to fetch analytics', e));
        }
    }, [view]);

    // ── Auth handlers ────────────────────────────────────────────────────────
    const handleAuthSuccess = (data) =>
        setCurrentUser({ user_id: data.user_id, display_name: data.display_name,
                         email: data.email, role: data.role });

    const handleLogout = async () => {
        try { await authAPI.logout(); } catch(e) {}
        setCurrentUser(null);
        setMessages([]);
        setActiveConversationId(null);
        setCurrentAnalysis(null);
        if (socketRef.current) { socketRef.current.close(); socketRef.current = null; }
    };

    const onCreateChat = async (targetId) => {
        const convId = await handleCreateChat(targetId);
        if (convId) {
            setActiveConversationId(convId);
            setMessages([]);
            setCurrentAnalysis(null);
            setActiveTab('chats');
        }
    };

    const onCreateGroup = async (name, memberIds) => {
        const convId = await handleCreateGroup(name, memberIds);
        if (convId) {
            setShowGroupModal(false);
            setActiveConversationId(convId);
            setMessages([]);
            setCurrentAnalysis(null);
            setActiveTab('chats');
        }
    };

    const onAddMember = async (userId) => {
        setMemberActionError('');
        try {
            await handleAddMember(activeConversationId, userId);
        } catch (e) {
            setMemberActionError(e.apiMessage || e.message || 'Failed to add member.');
        }
    };

    const onRemoveMember = async (userId) => {
        setMemberActionError('');
        try {
            await handleRemoveMember(activeConversationId, userId);
        } catch (e) {
            setMemberActionError(e.apiMessage || e.message || 'Failed to remove member.');
        }
    };

    const handleSendMessage = () => { sendMessage(); setView('live'); };

    // Members not yet in the active group (for the add-member picker)
    const currentMemberIds = activeConv?.members?.map(m => m.user_id) || [];
    const addableUsers = globalUsers.filter(u => !currentMemberIds.includes(u.user_id));

    // ── Render ───────────────────────────────────────────────────────────────
    return (
        <div className="app-shell">
            {/* Auth gate */}
            {!currentUser && sessionChecked && <LoginModal onSuccess={handleAuthSuccess} />}

            {/* Group creation modal */}
            {showGroupModal && (
                <GroupModal
                    globalUsers={globalUsers}
                    currentUserId={currentUser?.user_id}
                    onConfirm={onCreateGroup}
                    onClose={() => setShowGroupModal(false)}
                />
            )}

            {/* Loading gate */}
            {currentUser && !systemReady && (
                <div className={`loading-gate ${gateVisible ? '' : 'gate-exit'}`}>
                    <div className="gate-backdrop" />
                    <div className="gate-content">
                        <div className="brain-loader">
                            <div className="pulse-ring" />
                            <div className="pulse-ring delay" />
                            <div className="brain-icon">🧠</div>
                        </div>
                        <h2 className="gate-title">InnerLink is Initializing</h2>
                        <p className="gate-subtitle">Warming up neural pipelines and calibrating the meta-learner...</p>
                        <div className="gate-checklist">
                            {[
                                { key: 'redis',       label: 'Redis Stream Engine'      },
                                { key: 'database',    label: 'PostgreSQL Database'       },
                                { key: 'meta_learner',label: 'Meta-Learner Intelligence' }
                            ].map(({ key, label }) => (
                                <div key={key} className={`gate-check-item ${componentStatus[key] ? 'ready' : 'pending'}`}>
                                    <span className="check-icon">{componentStatus[key] ? '✓' : '◌'}</span>
                                    <span className="check-label">{label}</span>
                                </div>
                            ))}
                        </div>
                        <div className="gate-footer">
                            <div className="gate-spinner" />
                            <span>Checking every 3 seconds...</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Training banner */}
            {systemReady && trainingInProgress && (
                <div style={{
                    position: 'fixed', top: 0, left: 0, right: 0, zIndex: 500,
                    background: 'linear-gradient(90deg, #b45309, #d97706)',
                    color: '#fff', padding: '8px 20px',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    fontSize: '0.85rem', fontWeight: 500
                }}>
                    <span>🧠 <strong>AI is still learning</strong> — The meta-learner is training on new data.</span>
                    <button onClick={() => setTrainingInProgress(false)}
                            style={{ background: 'rgba(255,255,255,0.2)', border: 'none',
                                     color: '#fff', borderRadius: '4px', padding: '2px 10px',
                                     cursor: 'pointer' }}>✕ Dismiss</button>
                </div>
            )}

            <div className="bg-gradient" />
            <div className="glow-orb" id="orb-1" />
            <div className="glow-orb" id="orb-2" />

            {/* Sidebar */}
            <aside className="sidebar glass">
                <header className="app-header">
                    <div className="logo">
                        <div className="logo-icon" />
                        <h1>InnerLink</h1>
                    </div>
                    <div className="status-indicator">
                        <span className="pulse" /> {status}
                    </div>
                </header>

                {currentUser && (
                    <div style={{ padding: '0 20px', marginBottom: '10px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <h3 style={{ color: 'var(--accent-primary)', marginBottom: '10px' }}>
                                {currentUser.role === 'admin' && <span title="Admin" style={{ marginRight: '6px' }}>👑</span>}
                                {currentUser.display_name}
                            </h3>
                            <button id="logout-btn" onClick={handleLogout}
                                    style={{ background: 'rgba(255,80,80,0.15)', border: '1px solid rgba(255,80,80,0.2)',
                                             color: '#ff7070', borderRadius: '6px', padding: '4px 10px', cursor: 'pointer' }}>
                                Log out
                            </button>
                        </div>

                        <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                            {['chats', 'users'].map(tab => (
                                <button key={tab} onClick={() => setActiveTab(tab)}
                                        style={{ flex: 1, padding: '5px', border: '1px solid rgba(255,255,255,0.1)',
                                                 color: 'white', borderRadius: '5px', textTransform: 'capitalize',
                                                 background: activeTab === tab ? 'rgba(255,255,255,0.2)' : 'transparent',
                                                 cursor: 'pointer' }}>
                                    {tab}
                                </button>
                            ))}
                        </div>

                        {/* Users tab — start a direct chat */}
                        {activeTab === 'users' && (
                            <div className="users-list" style={{ maxHeight: '150px', overflowY: 'auto',
                                                                  marginBottom: '10px', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '5px' }}>
                                {globalUsers.map(u => (
                                    <div key={u.user_id} onClick={() => onCreateChat(u.user_id)}
                                         style={{ padding: '8px', cursor: 'pointer', borderBottom: '1px solid rgba(255,255,255,0.05)',
                                                  display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <span>{u.display_name}</span>
                                        <span style={{ color: 'var(--accent-primary)', fontSize: '1.1rem' }}>+</span>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Chats tab — conversation list */}
                        {activeTab === 'chats' && (
                            <>
                                {/* New Group button */}
                                <button
                                    onClick={() => setShowGroupModal(true)}
                                    style={{ width: '100%', marginBottom: '8px', padding: '6px',
                                             background: 'rgba(0,180,255,0.12)', border: '1px solid rgba(0,180,255,0.25)',
                                             color: 'var(--accent-primary)', borderRadius: '5px', cursor: 'pointer',
                                             fontSize: '0.85rem', fontWeight: 500 }}>
                                    + New Group Chat
                                </button>

                                <div className="chats-list" style={{ maxHeight: '150px', overflowY: 'auto',
                                                                      marginBottom: '10px', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '5px' }}>
                                    {conversations.length === 0 && (
                                        <div style={{ padding: '12px', color: '#64748b', fontSize: '0.85rem', textAlign: 'center' }}>
                                            No conversations yet
                                        </div>
                                    )}
                                    {conversations.map(c => (
                                        <div key={c.conversation_id}
                                             onClick={() => { setActiveConversationId(c.conversation_id); setMemberPanelOpen(false); }}
                                             style={{ padding: '8px', cursor: 'pointer',
                                                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                                                      background: activeConversationId === c.conversation_id ? 'rgba(0,180,255,0.3)' : 'transparent' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                {c.type === 'group' && (
                                                    <span style={{ fontSize: '0.7rem', background: 'rgba(0,180,255,0.2)',
                                                                   color: 'var(--accent-primary)', borderRadius: '3px',
                                                                   padding: '1px 5px', fontWeight: 600 }}>GROUP</span>
                                                )}
                                                <span style={{ fontWeight: 'bold' }}>
                                                    {c.type === 'group' ? c.name : c.other_display_name}
                                                </span>
                                                {c.type === 'group' && (
                                                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                                                        {c.member_count} members
                                                    </span>
                                                )}
                                            </div>
                                            <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '4px' }}>
                                                Vibe: <span style={{ color: EMOTION_COLORS[c.dominant_emotion?.toLowerCase()] || '#00f2ff' }}>
                                                    {c.dominant_emotion || 'Neutral'}
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </>
                        )}
                    </div>
                )}

                <div className="nav-links">
                    {[
                        { id: 'live',      icon: '⚡', label: 'Live Analysis'  },
                        { id: 'history',   icon: '📜', label: 'History & Trends' },
                        { id: 'analytics', icon: '📊', label: 'System Insights' }
                    ].map(({ id, icon, label }) => (
                        <button key={id} className={`nav-btn ${view === id ? 'active' : ''}`}
                                onClick={() => setView(id)}>
                            <span className="btn-icon">{icon}</span> {label}
                        </button>
                    ))}
                    {currentUser?.role === 'admin' && (
                        <button id="nav-admin-btn" className={`nav-btn ${view === 'admin' ? 'active' : ''}`}
                                onClick={() => setView('admin')}>
                            <span className="btn-icon">👑</span> Admin
                        </button>
                    )}
                </div>

                {/* No conversation selected — placeholder */}
                {!activeConversationId && (
                    <div className="empty-state" style={{ flex: 1, display: 'flex', flexDirection: 'column',
                                                          alignItems: 'center', justifyContent: 'center',
                                                          padding: '20px', textAlign: 'center' }}>
                        <div className="empty-icon">💬</div>
                        <p style={{ color: '#64748b', fontSize: '0.9rem', marginTop: '8px' }}>
                            Select a conversation or start a new one
                        </p>
                    </div>
                )}

                {/* Message list — only when a conversation is active */}
                {activeConversationId && (
                    <div className="chat-container" ref={chatContainerRef}>
                        {messages.length === 0 && (
                            <div className="empty-state">
                                <div className="empty-icon">✨</div>
                                <p>Send a message to start the analysis</p>
                            </div>
                        )}
                        {messages.map(msg => (
                            <div key={msg.id} className={`chat-msg ${msg.sender}`}
                                 data-sender={msg.senderName}
                                 style={{ position: 'relative' }}>
                                {/* Sender name in group chats */}
                                {isGroup && msg.sender !== 'user' && (
                                    <div style={{ fontSize: '0.7rem', color: 'var(--accent-primary)',
                                                  fontWeight: 600, marginBottom: '3px', opacity: 0.85 }}>
                                        {msg.senderName}
                                    </div>
                                )}
                                <div onClick={() => handleHistoryClick(msg)}
                                     style={{ cursor: msg.analysis ? 'pointer' : 'default', opacity: msg.analysis ? 1 : 0.8 }}
                                     title={msg.analysis ? 'Click to view emotional analysis' : 'Analysis pending...'}>
                                    {msg.text}
                                </div>
                                {msg.analysis && (
                                    <div style={{ fontSize: '0.75rem', marginTop: '6px', opacity: 0.9 }}>
                                        <span style={{ color: EMOTION_COLORS[msg.analysis.data.final_dominant_emotion?.toLowerCase()] || '#00f2ff', fontWeight: 'bold' }}>
                                            {msg.analysis.data.final_dominant_emotion}
                                        </span>
                                    </div>
                                )}
                                {/* Delete button — own messages only */}
                                {msg.sender === 'user' && (
                                    <button
                                        onClick={() => handleDeleteMessage(msg.id)}
                                        title="Delete message"
                                        className="msg-delete-btn">
                                        ✕
                                    </button>
                                )}
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                )}

                {/* Input — only when a conversation is active */}
                {activeConversationId && (
                    <div className="input-wrapper glass" style={{ position: 'relative' }}>
                        {showEmojiPicker && (
                            <div className="emoji-popup glass" style={{
                                position: 'absolute', bottom: '100%', left: 0, marginBottom: '10px',
                                padding: '12px', display: 'grid', gridTemplateColumns: 'repeat(6,1fr)',
                                gap: '8px', zIndex: 1000, background: 'rgba(17,25,40,0.95)'
                            }}>
                                {['😂','😭','😍','🔥','💀','🙃','🤔','🙄','💩','❤️','✨'].map(emoji => (
                                    <button key={emoji}
                                            onClick={() => { setInputValue(p => p + emoji); setShowEmojiPicker(false); }}
                                            style={{ background: 'none', border: 'none', fontSize: '1.5rem', cursor: 'pointer' }}>
                                        {emoji}
                                    </button>
                                ))}
                            </div>
                        )}
                        <button onClick={() => setShowEmojiPicker(!showEmojiPicker)}
                                style={{ background: 'none', border: 'none', fontSize: '1.5rem',
                                         cursor: 'pointer', opacity: 0.8, padding: '0 8px' }}>
                            😊
                        </button>
                        <textarea rows="1" placeholder="Type something expressive..."
                                  value={inputValue}
                                  onChange={e => setInputValue(e.target.value)}
                                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }} />
                        <button className="send-btn" onClick={handleSendMessage}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="22" y1="2" x2="11" y2="13" />
                                <polygon points="22 2 15 22 11 13 2 9 22 2" />
                            </svg>
                        </button>
                    </div>
                )}
            </aside>

            {/* Main dashboard */}
            <main className="dashboard">
                <div className="dashboard-header">
                    <div>
                        <h2>
                            {activeConv
                                ? isGroup
                                    ? activeConv.name
                                    : `Chat with ${activeConv.other_display_name}`
                                : 'Discovery Module'}
                        </h2>
                        {isGroup && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '4px' }}>
                                <span style={{ fontSize: '0.8rem', color: '#64748b' }}>
                                    {activeConv.member_count} members
                                </span>
                                <button
                                    onClick={() => setMemberPanelOpen(p => !p)}
                                    style={{ fontSize: '0.78rem', background: 'rgba(255,255,255,0.08)',
                                             border: '1px solid rgba(255,255,255,0.1)', color: '#94a3b8',
                                             borderRadius: '4px', padding: '2px 8px', cursor: 'pointer' }}>
                                    {memberPanelOpen ? 'Hide Members' : 'Manage Members'}
                                </button>
                            </div>
                        )}
                    </div>
                    <div className="analysis-meta">
                        {currentAnalysis ? `ID: ${currentAnalysis.data.id?.split('-')[0]}` : 'Ready'}
                    </div>
                </div>

                {/* Group members panel */}
                {isGroup && memberPanelOpen && (
                    <div className="glass" style={{ margin: '0 24px 16px', padding: '16px', borderRadius: '8px' }}>
                        <h4 style={{ marginBottom: '10px', color: 'var(--accent-primary)' }}>Members</h4>
                        {memberActionError && (
                            <p style={{ color: '#ff7070', fontSize: '0.85rem', marginBottom: '8px' }}>{memberActionError}</p>
                        )}
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '14px' }}>
                            {activeConv.members?.map(m => (
                                <div key={m.user_id} style={{ display: 'flex', alignItems: 'center', gap: '6px',
                                                              background: 'rgba(255,255,255,0.07)', borderRadius: '20px',
                                                              padding: '4px 12px', fontSize: '0.85rem' }}>
                                    <span>{m.display_name}</span>
                                    {m.user_id === activeConv.creator_user_id && (
                                        <span title="Creator" style={{ fontSize: '0.7rem', color: 'var(--accent-primary)' }}>★</span>
                                    )}
                                    {isCreator && m.user_id !== currentUser.user_id && (
                                        <button
                                            onClick={() => onRemoveMember(m.user_id)}
                                            style={{ background: 'none', border: 'none', color: '#ff7070',
                                                     cursor: 'pointer', fontSize: '0.8rem', padding: '0 2px' }}
                                            title="Remove member">✕</button>
                                    )}
                                </div>
                            ))}
                        </div>
                        {isCreator && addableUsers.length > 0 && (
                            <div>
                                <h5 style={{ marginBottom: '8px', color: '#94a3b8', fontSize: '0.85rem' }}>Add member</h5>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                    {addableUsers.map(u => (
                                        <button key={u.user_id}
                                                onClick={() => onAddMember(u.user_id)}
                                                style={{ background: 'rgba(0,180,255,0.1)', border: '1px solid rgba(0,180,255,0.2)',
                                                         color: 'var(--accent-primary)', borderRadius: '20px',
                                                         padding: '3px 12px', cursor: 'pointer', fontSize: '0.8rem' }}>
                                            + {u.display_name}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {view === 'live' && (
                    <LiveView currentAnalysis={currentAnalysis} vibeAnalysis={vibeAnalysis}
                              messages={messages} handleFeedback={handleFeedback} />
                )}
                {view === 'history' && (
                    <HistoryView messages={messages} handleFeedback={handleFeedback}
                                 handleHistoryClick={msg => { handleHistoryClick(msg); setView('live'); }} />
                )}
                {view === 'analytics' && <AnalyticsView analyticsData={analyticsData} />}
                {view === 'admin' && currentUser?.role === 'admin' && (
                    <div className="analytics-view" style={{ padding: '24px' }}>
                        <AdminDashboard currentUser={currentUser} />
                    </div>
                )}
            </main>
        </div>
    );
}

export default App;
