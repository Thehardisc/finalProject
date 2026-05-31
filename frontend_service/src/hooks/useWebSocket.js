import { useEffect, useRef } from 'react';
import { WS_BASE } from '../api/client';

/**
 * useWebSocket
 * Manages the WebSocket connection lifecycle, message dispatching,
 * and exponential-backoff auto-reconnect (F1).
 *
 * The connection is per-user (/ws/{user_id}), not per-conversation.
 * activeConversationId is intentionally excluded from the effect deps
 * so that switching conversations does not trigger a reconnect.
 */
export function useWebSocket({
    currentUser, activeConversationId, systemReady,
    setStatus, setMessages, setCurrentAnalysis, setVibeAnalysis,
    applyTheme, fetchVibe, socketRef
}) {
    const retryCountRef  = useRef(0);
    const retryTimerRef  = useRef(null);
    const mountedRef     = useRef(true);

    // Stable refs for callbacks that must stay current inside the long-lived
    // WebSocket handlers without causing a reconnect on every render.
    const applyThemeRef  = useRef(applyTheme);
    const fetchVibeRef   = useRef(fetchVibe);
    const currentUserRef = useRef(currentUser);
    useEffect(() => { applyThemeRef.current  = applyTheme;   }, [applyTheme]);
    useEffect(() => { fetchVibeRef.current   = fetchVibe;    }, [fetchVibe]);
    useEffect(() => { currentUserRef.current = currentUser;  }, [currentUser]);

    useEffect(() => {
        mountedRef.current = true;
        retryCountRef.current = 0;

        if (!currentUser || !systemReady) return;

        const MAX_RETRIES = 8;
        const BASE_DELAY  = 1000; // ms

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
                    const user = currentUserRef.current;

                    if (payload.type === 'analysis') {
                        setCurrentAnalysis(prev => ({
                            ...payload,
                            ai_insight:       null,
                            loadingReasoning: true
                        }));
                        applyThemeRef.current(payload.data.final_dominant_emotion);

                        if (payload.vibe) setVibeAnalysis(payload.vibe);
                        else fetchVibeRef.current();

                        setMessages(prev => {
                            const isSelf     = payload.data.sender_id === user?.user_id;
                            const newMsgData = {
                                id:         payload.data.id,
                                sender:     isSelf ? 'user' : 'ai',
                                text:       payload.data.raw_text,
                                timestamp:  payload.data.timestamp,
                                senderName: isSelf ? user?.display_name : payload.data.sender_id,
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

            // F1: exponential backoff reconnect
            ws.onclose = (event) => {
                if (!mountedRef.current) return;
                setStatus('Offline');

                // Don't retry if explicitly closed (code 1000) or policy violation (1008)
                if (event.code === 1000 || event.code === 1008) return;

                if (retryCountRef.current < MAX_RETRIES) {
                    const delay = Math.min(BASE_DELAY * 2 ** retryCountRef.current, 30_000);
                    retryCountRef.current++;
                    setStatus(`Reconnecting (${retryCountRef.current}/${MAX_RETRIES})...`);
                    retryTimerRef.current = setTimeout(connect, delay);
                } else {
                    setStatus('Connection failed — please refresh');
                }
            };

            ws.onerror = () => {
                ws.close(); // triggers onclose → retry
            };
        }

        connect();

        return () => {
            mountedRef.current = false;
            clearTimeout(retryTimerRef.current);
            if (socketRef.current) {
                socketRef.current.close(1000, 'component unmounted');
            }
        };
    }, [currentUser, systemReady]); // activeConversationId excluded — WS is per-user, not per-conversation
}
