import React, { useRef, useEffect, useState } from 'react';
import { CrystalBubble } from '../glass/CrystalBubble';
import { blendEmotions } from '../components/EmotionPalette';

// The emotion RGB from the last message with analysis, or neutral indigo
function getInputRgb(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const bert = messages[i]?.analysis?.data?.bert_emotions;
    if (bert && Array.isArray(bert) && bert.length > 0) {
      const dict = {};
      bert.forEach(({ label, score }) => { dict[label] = score; });
      return blendEmotions(dict);
    }
  }
  return '88, 86, 214';
}

export default function ChatPage({
  currentUser,
  messages,
  inputValue,
  setInputValue,
  onSend,
  onBack,
  onGoToAnalytics,
  activeConversationId,
  conversations,
  status,
  currentAnalysis,
  onFeedback,
  allEmotions,
}) {
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const [showFeedback, setShowFeedback] = useState(false);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [activeConversationId]);

  const inputRgb = getInputRgb(messages);
  const conv = conversations.find(c => c.conversation_id === activeConversationId);
  const partnerName = conv?.other_display_name || 'Chat';

  const dominant = currentAnalysis?.data?.final_dominant_emotion;

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: 'radial-gradient(ellipse at 50% 0%, rgba(88,86,214,0.05) 0%, #ffffff 60%)',
    }}>
      {/* ── Chat Top Bar ─────────────────────────────────────────────────── */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '16px 24px',
        background: 'rgba(255,255,255,0.78)',
        backdropFilter: 'blur(24px) saturate(180%)',
        WebkitBackdropFilter: 'blur(24px) saturate(180%)',
        borderBottom: '1px solid rgba(255,255,255,0.92)',
        boxShadow: '0 2px 16px rgba(0,0,0,0.05)',
        zIndex: 10,
        position: 'sticky', top: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button onClick={onBack} style={{
            background: 'rgba(0,0,0,0.05)', border: 'none', borderRadius: 10,
            width: 36, height: 36, cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1rem', color: '#374151',
            transition: 'background 0.15s',
          }}
            onMouseOver={e => e.currentTarget.style.background = 'rgba(0,0,0,0.09)'}
            onMouseOut={e => e.currentTarget.style.background = 'rgba(0,0,0,0.05)'}
          >
            ←
          </button>
          <div>
            <div style={{ fontWeight: 700, fontSize: '1rem', color: '#1c1c2e' }}>
              {partnerName}
            </div>
            {dominant && (
              <div style={{
                fontSize: '0.73rem', color: `rgb(${inputRgb})`,
                mixBlendMode: 'color-burn', fontWeight: 600, letterSpacing: '0.04em',
              }}>
                {dominant}
              </div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* WS status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.78rem', color: '#9ca3af' }}>
            <div style={{
              width: 7, height: 7, borderRadius: '50%',
              background: status === 'Live' ? 'rgb(52,199,89)' : '#d1d5db',
              boxShadow: status === 'Live' ? '0 0 5px rgba(52,199,89,0.7)' : 'none',
            }} />
            {status}
          </div>
          <button onClick={onGoToAnalytics} style={{
            background: 'rgba(0,0,0,0.05)', border: 'none', borderRadius: 10,
            width: 36, height: 36, cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1rem', color: '#374151',
            transition: 'background 0.15s',
          }}
            title="View Insights"
            onMouseOver={e => e.currentTarget.style.background = 'rgba(0,0,0,0.09)'}
            onMouseOut={e => e.currentTarget.style.background = 'rgba(0,0,0,0.05)'}
          >
            ◎
          </button>
        </div>
      </header>

      {/* ── Message Stream ───────────────────────────────────────────────── */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px 24px',
        display: 'flex',
        flexDirection: 'column',
      }}>
        {messages.length === 0 && (
          <div style={{
            flex: 1, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            color: '#9ca3af', fontSize: '0.9rem', gap: 10,
          }}>
            <div style={{
              width: 56, height: 56, borderRadius: 18,
              background: 'rgba(88,86,214,0.09)',
              border: '1.5px solid rgba(88,86,214,0.18)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '1.6rem',
            }}>
              💬
            </div>
            Start the conversation
          </div>
        )}

        {messages.map((msg, idx) => (
          <CrystalBubble
            key={msg.id ?? idx}
            message={msg}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* ── Glass Input Bar ──────────────────────────────────────────────── */}
      <div style={{
        padding: '14px 20px',
        background: 'rgba(255,255,255,0.78)',
        backdropFilter: 'blur(24px) saturate(180%)',
        WebkitBackdropFilter: 'blur(24px) saturate(180%)',
        borderTop: `1px solid rgba(${inputRgb}, 0.18)`,
        boxShadow: `0 -2px 20px rgba(${inputRgb}, 0.07)`,
        transition: 'border-color 0.4s ease, box-shadow 0.4s ease',
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12,
          background: `rgba(${inputRgb}, 0.07)`,
          border: `1.5px solid rgba(${inputRgb}, 0.22)`,
          borderRadius: 18,
          padding: '4px 8px 4px 16px',
          transition: 'border-color 0.4s ease, background 0.4s ease',
          boxShadow: `inset 0 1px 0 rgba(255,255,255,0.9), 0 2px 12px rgba(${inputRgb}, 0.10)`,
        }}>
          <textarea
            ref={inputRef}
            rows={1}
            placeholder="Say something..."
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                onSend();
              }
            }}
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              resize: 'none',
              fontSize: '0.96rem',
              lineHeight: 1.5,
              color: '#1c1c2e',
              fontFamily: 'inherit',
              padding: '8px 0',
              maxHeight: '120px',
              overflowY: 'auto',
            }}
          />
          <button
            onClick={onSend}
            disabled={!inputValue.trim()}
            style={{
              width: 40, height: 40,
              background: inputValue.trim()
                ? `linear-gradient(135deg, rgb(${inputRgb}), rgba(${inputRgb}, 0.75))`
                : 'rgba(0,0,0,0.07)',
              border: 'none', borderRadius: 12,
              cursor: inputValue.trim() ? 'pointer' : 'default',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'background 0.3s ease',
              boxShadow: inputValue.trim() ? `0 3px 12px rgba(${inputRgb}, 0.35)` : 'none',
              flexShrink: 0,
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
              stroke={inputValue.trim() ? '#ffffff' : '#9ca3af'}
              strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
