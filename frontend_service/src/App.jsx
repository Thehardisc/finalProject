import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import EmotionRadarChart from './components/EmotionRadarChart';
import MessageItem from './components/MessageItem';
import SystemLog from './components/SystemLog';
import Modal from './components/Modal';
import EmojiPicker from 'emoji-picker-react';

const API_URL = 'http://localhost:8001';

function App() {
    const [conversationId, setConversationId] = useState('conv-1');
    const [message, setMessage] = useState('');
    const [state, setState] = useState(null);
    const [lastEmotions, setLastEmotions] = useState({});
    const [history, setHistory] = useState([]);
    const [logs, setLogs] = useState([]); // System logs
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [showEmojiPicker, setShowEmojiPicker] = useState(false);

    const scrollContainerRef = useRef(null);
    const bottomRef = useRef(null);

    const addLog = (module, message, type = 'info') => {
        setLogs(prev => [...prev, {
            module,
            message,
            type,
            timestamp: new Date().toISOString()
        }].slice(-50)); // Keep last 50
    };


    useEffect(() => {
        const fetchMessages = async () => {
            try {
                // Ensure we use the correct port for API Service (8001)
                const response = await axios.get(`${API_URL}/conversation/${conversationId}/messages`);

                const messages = response.data.map(msg => ({
                    ...msg,
                    user_id: msg.sender_id,
                    emotions: typeof msg.emotions === 'string' ? JSON.parse(msg.emotions) : msg.emotions,
                    reasoning: msg.reasoning ? (typeof msg.reasoning === 'string' ? JSON.parse(msg.reasoning) : msg.reasoning) : null,
                    pipeline_log: msg.pipeline_log ? (typeof msg.pipeline_log === 'string' ? JSON.parse(msg.pipeline_log) : msg.pipeline_log) : null
                }));

                setHistory(messages.reverse());

                if (messages.length > 0) {
                    const latest = messages[messages.length - 1]; // Newest
                    setLastEmotions(latest.emotions || {});
                }

            } catch (err) {
                console.error("Polling error:", err);
            }
        };

        fetchMessages();
        const interval = setInterval(fetchMessages, 2000);
        return () => clearInterval(interval);
    }, [conversationId]);

    // Smart Auto-scroll: Only scroll if we were already at the bottom OR if it's the first load
    useEffect(() => {
        if (history.length === 0) return;

        const container = scrollContainerRef.current;
        if (!container) return;

        // threshold for "at bottom"
        const isAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 150;

        if (isAtBottom || history.length < 5) {
            // Use requestAnimationFrame to ensure DOM is updated
            requestAnimationFrame(() => {
                bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
            });
        }
    }, [history.length, history[history.length - 1]?.id]);

    const sendMessage = async (e) => {
        e.preventDefault();
        if (!message.trim()) return;



        try {
            await axios.post('http://localhost:8000/messages', {
                user_id: 'user-1',
                conversation_id: conversationId,
                text: message,
            });
            setMessage('');
            addLog('INGESTION', 'Message accepted by Ingestion Service.', 'success');
        } catch (err) {
            console.error(err);
            alert(`Failed to send: ${err.message}`);
            addLog('ERROR', `Failed to send: ${err.message}`, 'error');
        }
    };

    return (
        <div style={{
            display: 'flex',
            width: '100vw',
            height: '100vh',
            backgroundColor: '#000000',
            color: '#a1a1aa',
            overflow: 'hidden'
        }}>



            <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
                height: '100%',
                overflow: 'hidden'
            }}>

                <div style={{
                    padding: '1rem 2rem',
                    borderBottom: '1px solid #27272a',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: 'rgba(9, 9, 11, 0.8)',
                    backdropFilter: 'blur(10px)',
                    zIndex: 30
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: 'white' }}>Emotion AI Chat</div>
                        <div style={{
                            padding: '2px 8px', borderRadius: '4px',
                            background: '#27272a', fontSize: '0.7rem', color: '#a1a1aa'
                        }}>
                            ID: {conversationId}
                        </div>
                    </div>

                    <button
                        onClick={() => setIsModalOpen(true)}
                        style={{
                            padding: '0.5rem 1rem',
                            borderRadius: '0.5rem',
                            background: '#3f3f46',
                            color: 'white',
                            border: 'none',
                            cursor: 'pointer',
                            fontSize: '0.85rem',
                            fontWeight: '500',
                            transition: 'background 0.2s',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem'
                        }}>
                        <span>📊</span> View Neural Model
                    </button>
                </div>


                <div
                    ref={scrollContainerRef}
                    style={{
                        flex: 1,
                        display: 'flex',
                        flexDirection: 'column',
                        overflowY: 'auto',
                        minHeight: 0,
                        scrollBehavior: 'smooth'
                    }}
                >
                    <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column' }}>

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


                <div style={{ padding: '2rem', flexShrink: 0, borderTop: '1px solid #27272a', background: '#09090b', zIndex: 20 }}>
                    <form onSubmit={sendMessage} style={{ display: 'flex', gap: '1rem', maxWidth: '900px', margin: '0 auto', position: 'relative' }}>

                        <div style={{ display: 'flex', alignItems: 'center' }}>
                            <button
                                type="button"
                                onClick={() => setShowEmojiPicker(!showEmojiPicker)}
                                style={{
                                    background: 'transparent',
                                    border: 'none',
                                    fontSize: '1.5rem',
                                    cursor: 'pointer',
                                    padding: '0 0.5rem',
                                    filter: 'grayscale(100%) brightness(150%)',
                                    transition: 'filter 0.2s'
                                }}
                                onMouseOver={(e) => e.currentTarget.style.filter = 'none'}
                                onMouseOut={(e) => e.currentTarget.style.filter = 'grayscale(100%) brightness(150%)'}
                            >
                                😊
                            </button>

                            {showEmojiPicker && (
                                <div style={{ position: 'absolute', bottom: '80px', left: 0, zIndex: 100 }}>
                                    <EmojiPicker
                                        theme="dark"
                                        onEmojiClick={(emojiData) => {
                                            setMessage(prev => prev + emojiData.emoji);
                                        }}
                                    />
                                </div>
                            )}
                        </div>

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


            <div style={{
                width: '320px', minWidth: '320px', flexShrink: 0,
                borderLeft: '1px solid #27272a',
                background: 'black', zIndex: 10,
                height: '100%'
            }}>
                <SystemLog logs={logs} />
            </div>


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
