import React, { useState } from 'react';
import { EmotionPalette } from '../components/EmotionPalette';

// Resolve emotion name → RGB string for card glow
const emotionRgb = (emo) => EmotionPalette[emo?.toLowerCase()] || EmotionPalette.neutral;

export default function DashboardPage({
  currentUser,
  conversations,
  globalUsers,
  activeConversationId,
  onSelectConversation,
  onCreateChat,
  onGoToAnalytics,
  onLogout,
  status,
}) {
  const [showUserPicker, setShowUserPicker] = useState(false);
  const [blurIntensity, setBlurIntensity] = useState(24);
  const [showSettings, setShowSettings] = useState(false);

  const handlePickUser = (userId) => {
    onCreateChat(userId);
    setShowUserPicker(false);
  };

  // Dynamic --bubble-rgb for prism button tint
  const accentRgb = '0, 119, 255';

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* ── Top bar ─────────────────────────────────────────────────────── */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '20px 32px 0',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: 38, height: 38, borderRadius: 11,
            background: 'linear-gradient(135deg, #0077ff 0%, #7000ff 100%)',
            boxShadow: '0 4px 16px rgba(0,119,255,0.32)',
          }} />
          <span style={{ fontSize: '1.4rem', fontWeight: 800, color: '#1c1c2e', letterSpacing: '-0.5px' }}>
            InnerLink
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Status dot */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: '#6b7280' }}>
            <div style={{
              width: 8, height: 8, borderRadius: '50%',
              background: status === 'Live' ? 'rgb(52, 199, 89)' : '#d1d5db',
              boxShadow: status === 'Live' ? '0 0 6px rgba(52,199,89,0.6)' : 'none',
            }} />
            {status}
          </div>

          <button
            className="btn-tab"
            style={{ '--bubble-rgb': accentRgb }}
            onClick={() => setShowSettings(s => !s)}
          >
            Settings
          </button>

          {currentUser?.role === 'admin' && (
            <button
              className="btn-tab"
              style={{ '--bubble-rgb': accentRgb, fontSize: '0.8rem' }}
              onClick={onGoToAnalytics}
            >
              Admin
            </button>
          )}

          <div style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '6px 14px',
            background: 'rgba(255,255,255,0.75)',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
            border: '1px solid rgba(255,255,255,0.9)',
            borderRadius: 12,
            fontSize: '0.85rem', fontWeight: 600, color: '#1c1c2e',
            boxShadow: '0 2px 10px rgba(0,0,0,0.06)',
          }}>
            {currentUser?.role === 'admin' && <span style={{ fontSize: '0.75rem' }}>👑</span>}
            {currentUser?.display_name}
            <button onClick={onLogout} style={{
              background: 'rgba(255, 60, 60, 0.10)',
              border: '1px solid rgba(255,60,60,0.18)',
              color: '#cc2222', borderRadius: 7, padding: '2px 10px',
              cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600, marginLeft: 4,
            }}>
              Log out
            </button>
          </div>
        </div>
      </header>

      {/* ── Settings Panel ───────────────────────────────────────────────── */}
      {showSettings && (
        <div style={{
          margin: '16px 32px 0',
          padding: '18px 24px',
          background: 'rgba(255,255,255,0.80)',
          backdropFilter: `blur(${blurIntensity}px) saturate(180%)`,
          WebkitBackdropFilter: `blur(${blurIntensity}px) saturate(180%)`,
          border: '1px solid rgba(255,255,255,0.92)',
          borderRadius: 16,
          boxShadow: '0 4px 20px rgba(0,0,0,0.06)',
          animation: 'fadeUp 0.28s cubic-bezier(0.34,1.2,0.64,1) both',
          display: 'flex', alignItems: 'center', gap: 20,
        }}>
          <span style={{ fontSize: '0.84rem', fontWeight: 600, color: '#374151' }}>Glass Blur</span>
          <input
            type="range" min={8} max={40} step={2}
            value={blurIntensity}
            onChange={e => setBlurIntensity(Number(e.target.value))}
            style={{ flex: 1, accentColor: '#0077ff' }}
          />
          <span style={{ fontSize: '0.82rem', color: '#6b7280', minWidth: 40 }}>{blurIntensity}px</span>
        </div>
      )}

      {/* ── Feed ────────────────────────────────────────────────────────── */}
      <main style={{ flex: 1, padding: '24px 32px 40px' }}>
        {/* Section header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700, color: '#1c1c2e', letterSpacing: '-0.2px' }}>
              Conversations
            </h2>
            <p style={{ margin: '2px 0 0', fontSize: '0.82rem', color: '#6b7280' }}>
              {conversations.length} active thread{conversations.length !== 1 ? 's' : ''}
            </p>
          </div>

          <button
            className="btn-prism"
            style={{ '--bubble-rgb': accentRgb }}
            onClick={() => setShowUserPicker(p => !p)}
          >
            + New Chat
          </button>
        </div>

        {/* User picker overlay */}
        {showUserPicker && (
          <div style={{
            marginBottom: 20,
            padding: '16px 20px',
            background: 'rgba(255,255,255,0.88)',
            backdropFilter: 'blur(20px) saturate(180%)',
            WebkitBackdropFilter: 'blur(20px) saturate(180%)',
            border: '1px solid rgba(255,255,255,0.95)',
            borderRadius: 16,
            boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
            animation: 'fadeUp 0.28s cubic-bezier(0.34,1.2,0.64,1) both',
          }}>
            <p style={{ margin: '0 0 12px', fontSize: '0.82rem', fontWeight: 600, color: '#374151', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Start chat with
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {globalUsers.length === 0 && (
                <p style={{ color: '#9ca3af', fontSize: '0.85rem', margin: 0 }}>No other users yet.</p>
              )}
              {globalUsers.map(u => (
                <button key={u.user_id} onClick={() => handlePickUser(u.user_id)} style={{
                  background: 'rgba(0,119,255,0.06)',
                  border: '1px solid rgba(0,119,255,0.14)',
                  borderRadius: 10, padding: '10px 14px',
                  textAlign: 'left', cursor: 'pointer',
                  fontSize: '0.9rem', fontWeight: 500, color: '#1c1c2e',
                  transition: 'background 0.15s',
                }}>
                  {u.display_name}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Conversation cards grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
          {conversations.length === 0 && (
            <div style={{
              gridColumn: '1 / -1',
              padding: '48px 24px',
              textAlign: 'center',
              color: '#9ca3af',
              fontSize: '0.9rem',
            }}>
              No conversations yet — click <strong>+ New Chat</strong> to begin.
            </div>
          )}
          {conversations.map((conv, i) => {
            const rgb = emotionRgb(conv.dominant_emotion);
            const isActive = conv.conversation_id === activeConversationId;
            return (
              <div
                key={conv.conversation_id}
                className={`conv-card ${isActive ? 'active' : ''}`}
                style={{
                  '--card-rgb': rgb,
                  animationDelay: `${i * 0.05}s`,
                }}
                onClick={() => onSelectConversation(conv.conversation_id)}
              >
                {/* Partner initial */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                  <div style={{
                    width: 44, height: 44, borderRadius: 14,
                    background: `rgba(${rgb}, 0.15)`,
                    border: `1.5px solid rgba(${rgb}, 0.30)`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '1.1rem', fontWeight: 800,
                    color: `rgb(${rgb})`,
                    mixBlendMode: 'color-burn',
                    boxShadow: `0 2px 10px rgba(${rgb}, 0.18)`,
                  }}>
                    {conv.other_display_name?.[0]?.toUpperCase() || '?'}
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#1c1c2e' }}>
                      {conv.other_display_name}
                    </div>
                    <div style={{ fontSize: '0.76rem', color: '#6b7280', marginTop: 1 }}>
                      Valence {(conv.average_valence || 0).toFixed(2)}
                    </div>
                  </div>
                </div>

                {/* Emotion badge + glow bar */}
                {conv.dominant_emotion && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{
                      height: 4, flex: 1, borderRadius: 99,
                      background: `linear-gradient(90deg, rgba(${rgb}, 0.7), rgba(${rgb}, 0.15))`,
                      boxShadow: `0 0 8px rgba(${rgb}, 0.35)`,
                    }} />
                    <span style={{
                      fontSize: '0.73rem', fontWeight: 700,
                      color: `rgb(${rgb})`,
                      mixBlendMode: 'color-burn',
                      textTransform: 'capitalize',
                      letterSpacing: '0.04em',
                    }}>
                      {conv.dominant_emotion}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Analytics shortcut */}
        {conversations.length > 0 && (
          <div style={{ marginTop: 32, textAlign: 'center' }}>
            <button
              className="btn-tab"
              style={{ '--bubble-rgb': accentRgb }}
              onClick={onGoToAnalytics}
            >
              View Insights Board
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
