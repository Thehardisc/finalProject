import { useState, useRef, useEffect } from 'react';
import { usersAPI, feedbackAPI } from '../api/client';


/**
 * useMessages
 * Manages the message list, sending, feedback, history loading, and analysis parsing.
 */
export function useMessages(currentUser, activeConversationId,
                             systemReady, conversations, globalUsers) {
    const [messages,        setMessages]        = useState([]);
    const [currentAnalysis, setCurrentAnalysis] = useState(null);
    const [vibeAnalysis,    setVibeAnalysis]    = useState(null);
    const [inputValue,      setInputValue]      = useState('');

    const socketRef     = useRef(null);
    const messagesEndRef = useRef(null);

    // ── Helpers ────────────────────────────────────────────────────────────────
    const getSenderName = (senderId) => {
        if (!senderId) return 'System';
        if (currentUser && senderId === currentUser.user_id) return currentUser.display_name;
        // Check direct conversation
        const direct = conversations.find(c => c.type !== 'group' && c.other_user_id === senderId);
        if (direct) return direct.other_display_name;
        // Check group member lists
        for (const conv of conversations) {
            if (conv.members) {
                const member = conv.members.find(m => m.user_id === senderId);
                if (member) return member.display_name;
            }
        }
        const gu = globalUsers.find(u => u.user_id === senderId);
        if (gu) return gu.display_name;
        return senderId.substring(0, 8);
    };

    const parseAnalysis = (msg) => {
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
                    id:                    msg.id,
                    raw_text:              msg.content || msg.text,
                    final_dominant_emotion: ems.dominant_emotion || 'Neutral',
                    final_valence:         ems.vader_compound || 0,
                    bert_emotions:         bert_list,
                    llm_insights:          'Detailed analysis loaded.',
                    llm_sarcasm_score:     0,
                    hierarchical_scores:   [],
                    emojis_found:          [],
                    slang_detected:        {}
                }
            };
        } catch (e) {
            console.error('Parse Error', e);
            return null;
        }
    };

    const fetchVibe = async () => {
        if (!activeConversationId) return;
        try {
            const stateRes = await usersAPI.conversationState(activeConversationId);
            if (stateRes.data) {
                setVibeAnalysis({
                    valence:     parseFloat(stateRes.data.average_valence || 0),
                    top_emotions:[stateRes.data.dominant_emotion || 'Neutral']
                });
            }
        } catch (e) { console.warn('Could not fetch latest vibe state'); }
    };

    // ── Send message ──────────────────────────────────────────────────────────
    const sendMessage = () => {
        if (!inputValue.trim()) return;
        const newMsg = {
            id:         Date.now(),
            sender:     'user',
            text:       inputValue,
            senderName: currentUser.display_name,
            analysis:   null
        };
        setMessages(prev => [...prev, newMsg]);
        if (socketRef.current && activeConversationId) {
            socketRef.current.send(JSON.stringify({
                text:            inputValue,
                recipient_id:    'system',
                sender_id:       currentUser.user_id,
                conversation_id: activeConversationId
            }));
        }
        setInputValue('');
    };

    // ── Feedback ──────────────────────────────────────────────────────────────
    const handleFeedback = async (msgId, newLabel) => {
        try {
            await feedbackAPI.post(msgId, newLabel);
            setMessages(prev => prev.map(m =>
                m.id === msgId ? { ...m, verified: true, feedbackLabel: newLabel } : m
            ));
            if (currentAnalysis?.data.id === msgId) {
                setCurrentAnalysis(prev => ({ ...prev, verified: true, feedbackLabel: newLabel }));
            }
        } catch (e) { console.error('Failed to send feedback', e); }
    };

    const handleHistoryClick = (msg) => {
        if (msg.analysis) {
            setCurrentAnalysis(msg.analysis);
        }
    };

    const handleDeleteMessage = async (msgId) => {
        try {
            await feedbackAPI.delete(msgId);
            setMessages(prev => prev.filter(m => m.id !== msgId));
            setCurrentAnalysis(prev =>
                prev?.data?.id === msgId ? null : prev
            );
        } catch (e) {
            console.error('Failed to delete message', e);
        }
    };

    // ── History fetch on conversation change ──────────────────────────────────
    useEffect(() => {
        const fetchInitialState = async () => {
            if (!activeConversationId) return;
            try {
                const res = await usersAPI.messages(activeConversationId, 50);
                if (res.data?.length > 0) {
                    const historyMsgs = res.data.slice().reverse().map(m => {
                        const parsed = parseAnalysis(m);
                        const isSelf = currentUser && m.sender_id === currentUser.user_id;
                        return {
                            id:         m.id,
                            sender:     isSelf ? 'user' : 'ai',
                            text:       m.content,
                            senderName: isSelf ? currentUser.display_name : getSenderName(m.sender_id),
                            analysis:   parsed
                        };
                    });
                    setMessages(historyMsgs);
                    const initialAnalysis = parseAnalysis(res.data[0]);
                    if (initialAnalysis) setCurrentAnalysis(initialAnalysis);
                    await fetchVibe();
                }
            } catch (err) { console.error('Initial fetch failed:', err); }
        };

        if (currentUser && systemReady) fetchInitialState();
    }, [activeConversationId, currentUser, systemReady]);

    // ── Auto-scroll ───────────────────────────────────────────────────────────
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return {
        messages, setMessages, currentAnalysis, setCurrentAnalysis,
        vibeAnalysis, setVibeAnalysis, inputValue, setInputValue,
        socketRef, messagesEndRef, fetchVibe, parseAnalysis, getSenderName,
        sendMessage, handleFeedback, handleHistoryClick, handleDeleteMessage
    };
}
