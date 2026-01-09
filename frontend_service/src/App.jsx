import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import EmotionRadarChart from './components/EmotionRadarChart';
import MessageItem from './components/MessageItem';
import SystemLog from './components/SystemLog';
import Modal from './components/Modal';

const API_URL = 'http://localhost:8001';

function App() {
    const [conversationId, setConversationId] = useState('conv-1');
    const [message, setMessage] = useState('');
    const [state, setState] = useState(null);
    const [lastEmotions, setLastEmotions] = useState({});
    const [history, setHistory] = useState([]);
    const [logs, setLogs] = useState([]); // System logs
    const [isModalOpen, setIsModalOpen] = useState(false);

    const bottomRef = useRef(null);

    // Add Log Helper
    const addLog = (module, message, type = 'info') => {
        const newLog = { timestamp: Date.now(), module, message, type };
        setLogs(prev => [...prev.slice(-49), newLog]); // Keep last 50
    };

    // Poll for data
    useEffect(() => {
        addLog('SYSTEM', 'Initializing connection to Cortex API...', 'info');

        const fetchData = async () => {
            try {
                const stateRes = await axios.get(`${API_URL}/conversation/${conversationId}/state`);
                setState(stateRes.data);
                if (stateRes.data.last_message_emotions) {
                    try { setLastEmotions(JSON.parse(stateRes.data.last_message_emotions)); } catch (e) { }
                }

                const historyRes = await axios.get(`${API_URL}/conversation/${conversationId}/messages`);
                const newHistory = historyRes.data.reverse();

                setHistory(prev => {
                    // Deep check: Length diff OR Last ID diff
                    const isDifferent = newHistory.length !== prev.length ||
                        (newHistory.length > 0 && newHistory[newHistory.length - 1].id !== prev[prev.length - 1]?.id);

                    if (isDifferent) {
                        return newHistory;
                    }
                    return prev;
                });
            } catch (err) {
                // console.error(err); 
                // addLog('ERROR', `Connection failed: ${err.message}`, 'error');
            }
        };

        fetchData(); // Initial fetch
        const interval = setInterval(fetchData, 2000);
        return () => clearInterval(interval);
    }, [conversationId]); // Re-run if ID changes

    // Auto-scroll to bottom of CHAT
    // Smart Auto-scroll: Only scroll if we were already at the bottom OR if it's the first load
    useEffect(() => {
        if (history.length === 0) return;

        const container = bottomRef.current?.parentElement;
        if (!container) return;

        // Threshold for "at bottom" (e.g., within 100px)
        const isAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 150;

        // Always scroll on first load, otherwise only if user was already at bottom
        if (isAtBottom || history.length < 5) {
            bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
    }, [history.length, history[history.length - 1]?.id]);

    const sendMessage = async (e) => {
        e.preventDefault();
        if (!message.trim()) return;
        addLog('USER_INPUT', `dispatching message payload: "${message.substring(0, 20)}..."`, 'warn');
        try {
            await axios.post('http://localhost:8000/messages', {
                user_id: 'user-1',
                conversation_id: conversationId,
                text: message,
                timestamp: new Date().toISOString()
            });
            setMessage('');
            addLog('INGESTION', 'Message accepted by Ingestion Service.', 'success');
        } catch (err) {
            alert(err.message);
            addLog('ERROR', `Failed to send: ${err.message}`, 'error');
        }
    };

    return (
        <div style={{
            display: 'flex', height: '100dvh', width: '100vw',
            position: 'relative',
            background: '#000000',
            color: '#e4e4e7',
            fontFamily: '"Inter", sans-serif',
            overflow: 'hidden'
        }}>

            {/* Background Ambient Glow */}
            <div style={{ position: 'absolute', top: '-20%', left: '-10%', width: '50%', height: '50%', background: 'radial-gradient(circle, rgba(59,130,246,0.15) 0%, rgba(0,0,0,0) 70%)', pointerEvents: 'none' }} />
            <div style={{ position: 'absolute', bottom: '-20%', right: '-10%', width: '50%', height: '50%', background: 'radial-gradient(circle, rgba(139,92,246,0.15) 0%, rgba(0,0,0,0) 70%)', pointerEvents: 'none' }} />

            {/* --- LEFT SIDEBAR (Navigation) --- */}
            <div style={{
                width: '280px', minWidth: '280px', flexShrink: 0,
                borderRight: '1px solid #27272a',
                display: 'flex', flexDirection: 'column',
                background: 'rgba(9, 9, 11, 0.8)', backdropFilter: 'blur(10px)',
                zIndex: 10,
                height: '100%'
            }}>
                <div style={{ padding: '2rem 1.5rem', borderBottom: '1px solid #27272a' }}>
                    <h1 style={{ margin: 0, fontSize: '1.1rem', letterSpacing: '0.05em', background: 'linear-gradient(to right, #60a5fa, #a78bfa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontWeight: '900' }}>
                        CORTEX // V1
                    </h1>
                    <div style={{ fontSize: '0.7rem', color: '#52525b', marginTop: '4px' }}>EMOTIONAL INTELLIGENCE UNIT</div>
                </div>

                <div style={{ flex: 1, overflowY: 'auto', padding: '1rem' }}>
                    <div style={{ fontSize: '0.7rem', color: '#71717a', marginBottom: '0.5rem', fontWeight: 'bold' }}>DETECTED SIGNALS</div>
                    <div
                        onClick={() => setConversationId('conv-1')}
                        style={{
                            padding: '1rem', borderRadius: '0.75rem', cursor: 'pointer', marginBottom: '0.8rem',
                            background: conversationId === 'conv-1' ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                            border: conversationId === 'conv-1' ? '1px solid #3b82f6' : '1px solid transparent',
                            transition: 'all 0.2s'
                        }}
                    >
                        <div style={{ fontWeight: 'bold', color: conversationId === 'conv-1' ? '#60a5fa' : 'inherit' }}># GLOBAL_CHAT</div>
                        <div style={{ fontSize: '0.75rem', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#4ade80' }}></span>
                            <span style={{ opacity: 0.6 }}>Active • {state?.message_count || 0} msgs</span>
                        </div>
                    </div>
                </div>

                {/* Vibe Check Button */}
                <div style={{ padding: '1.5rem', marginTop: 'auto' }}>
                    <button
                        onClick={() => setIsModalOpen(true)}
                        style={{
                            width: '100%', padding: '1rem', borderRadius: '0.75rem',
                            background: 'linear-gradient(135deg, #2563eb 0%, #4f46e5 100%)',
                            color: 'white', fontWeight: 'bold', border: 'none', cursor: 'pointer',
                            boxShadow: '0 4px 15px rgba(37, 99, 235, 0.4)',
                            transition: 'transform 0.1s',
                            textTransform: 'uppercase', letterSpacing: '0.05em'
                        }}
                        onMouseDown={e => e.currentTarget.style.transform = 'scale(0.98)'}
                        onMouseUp={e => e.currentTarget.style.transform = 'scale(1)'}
                    >
                        Open Neural Radar
                    </button>
                </div>
            </div>

            {/* --- CENTER (Chat Feed) --- */}
            <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
                height: '100%',
                overflow: 'hidden' // CRITICAL: Stop parent from scrolling
            }}>

                {/* Header */}
                <div style={{
                    height: '80px', flexShrink: 0, borderBottom: '1px solid #27272a',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 2rem',
                    background: 'rgba(9, 9, 11, 0.5)', backdropFilter: 'blur(5px)'
                }}>
                    <div>
                        <h2 style={{ margin: 0, fontSize: '1.2rem' }}>Global Uplink</h2>
                        <span style={{ fontSize: '0.8rem', color: '#71717a' }}>Last Sync: {state?.overall_mood || "SYNCING..."}</span>
                    </div>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        {/* Status Indicators */}
                        <div style={{ padding: '0.5rem 1rem', background: '#18181b', borderRadius: '2rem', border: '1px solid #27272a', fontSize: '0.75rem', color: '#a1a1aa' }}>
                            Latency: <span style={{ color: '#4ade80' }}>12ms</span>
                        </div>
                    </div>
                </div>

                {/* Messages Area - SCROLL FIX (Final) */}
                <div style={{
                    flex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    overflowY: 'auto',
                    minHeight: 0,
                    scrollBehavior: 'smooth'
                }}>
                    <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column' }}>
                        {/* Spacer for empty state */}
                        {history.length === 0 && (
                            <div style={{ margin: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', opacity: 0.3, padding: '4rem' }}>
                                <div style={{ fontSize: '3rem' }}>📡</div>
                                <p>Awaiting Transmission...</p>
                            </div>
                        )}

                        {history.map((msg) => (
                            <MessageItem key={msg.id} message={msg} isOwn={msg.user_id === 'user-1'} />
                        ))}
                        <div ref={bottomRef} style={{ height: '1px', flexShrink: 0 }} />
                    </div>
                </div>

                {/* Input Area */}
                <div style={{ padding: '2rem', flexShrink: 0, borderTop: '1px solid #27272a', background: '#09090b', zIndex: 20 }}>
                    <form onSubmit={sendMessage} style={{ display: 'flex', gap: '1rem', maxWidth: '900px', margin: '0 auto' }}>
                        <input
                            type="text"
                            placeholder="Enter command or message..."
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                            style={{
                                flex: 1, padding: '1rem 1.5rem', borderRadius: '1rem',
                                border: '1px solid #27272a',
                                background: '#18181b', color: 'white', outline: 'none',
                                boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.3)',
                                fontFamily: 'inherit'
                            }}
                        />
                        <button type="submit" style={{
                            padding: '0 2.5rem', borderRadius: '1rem',
                            background: '#27272a', color: 'white', fontWeight: 'bold',
                            border: '1px solid #3f3f46', cursor: 'pointer',
                            transition: 'all 0.2s'
                        }}>
                            SEND
                        </button>
                    </form>
                </div>

            </div>

            {/* --- RIGHT SIDEBAR (System Logs) --- */}
            <div style={{
                width: '320px', minWidth: '320px', flexShrink: 0,
                borderLeft: '1px solid #27272a',
                background: 'black', zIndex: 10,
                height: '100%'
            }}>
                <SystemLog logs={logs} />
            </div>

            {/* --- MODAL FOR RADAR --- */}
            <Modal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} title="Neural Reflection Model">
                <div style={{ height: '400px', width: '100%' }}>
                    <EmotionRadarChart emotionData={lastEmotions} />
                </div>
                <div style={{ marginTop: '1rem', textAlign: 'center', color: '#a1a1aa', fontSize: '0.8rem' }}>
                    Real-time analysis of the last processed message vector.
                </div>
            </Modal>

        </div>
    );
}

export default App;
