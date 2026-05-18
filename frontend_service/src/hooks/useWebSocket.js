import { useEffect, useRef } from 'react';
import { WS_BASE } from '../constants/emotions';

/**
 * useWebSocket
 * Manages the WebSocket connection lifecycle, message dispatching,
 * and auto-reconnect on conversation change.
 */
export function useWebSocket({
    currentUser, activeConversationId, systemReady,
    setStatus, setMessages, setCurrentAnalysis, setVibeAnalysis,
    applyTheme, fetchVibe, socketRef
}) {
    useEffect(() => {
        if (!currentUser || !systemReady) return;

        const ws = new WebSocket(`${WS_BASE}/ws/${currentUser.user_id}`);

        ws.onopen = () => {
            console.log('Connected to WS');
            setStatus('Live');
        };

        ws.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);

                if (payload.type === 'analysis') {
                    setCurrentAnalysis(prev => ({
                        ...payload,
                        ai_insight:      null,
                        loadingReasoning: true
                    }));
                    applyTheme(payload.data.final_dominant_emotion);

                    if (payload.vibe) setVibeAnalysis(payload.vibe);
                    else fetchVibe();

                    setMessages(prev => {
                        const isSelf     = payload.data.sender_id === currentUser?.user_id;
                        const newMsgData = {
                            id:         payload.data.id,
                            sender:     isSelf ? 'user' : 'ai',
                            text:       payload.data.raw_text,
                            senderName: isSelf ? currentUser?.display_name : payload.data.sender_id,
                            analysis:   payload
                        };

                        if (prev.some(m => m.id === payload.data.id)) {
                            return prev.map(m => m.id === payload.data.id
                                ? { ...m, analysis: payload } : m);
                        }

                        const existingIdx = prev.findIndex(
                            m => m.text === payload.data.raw_text && !m.analysis
                                && isSelf && m.sender === 'user'
                        );

                        if (existingIdx >= 0) {
                            const updated = [...prev];
                            updated[existingIdx] = { ...updated[existingIdx], ...newMsgData };
                            return updated;
                        }

                        return [...prev, newMsgData];
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
                console.error('WS Parse error', e);
            }
        };

        ws.onclose = () => setStatus('Offline');

        socketRef.current = ws;

        return () => {
            ws.close();
        };
    }, [activeConversationId, currentUser, systemReady]);
}
