import React, { useState, useRef, useEffect } from 'react';
import { EmotionPalette, blendEmotions } from '../components/EmotionPalette';
import TelemetryPanel  from '../components/TelemetryPanel';
import AnalysisDrawer  from '../components/AnalysisDrawer';
import DemoRunner      from '../components/DemoRunner';

// ── Helpers ───────────────────────────────────────────────────────────────────

const emotionRgb = (emo) => EmotionPalette[emo?.toLowerCase()] || EmotionPalette.neutral;

function timeAgo(id) {
  if (!id) return '';
  const ts = typeof id === 'number' ? id : parseInt(id, 10);
  if (isNaN(ts) || ts < 1e12) return '';
  const diff = Date.now() - ts;
  if (diff < 60000)    return 'now';
  if (diff < 3600000)  return `${Math.floor(diff / 60000)}m`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`;
  return `${Math.floor(diff / 86400000)}d`;
}

function bubbleGradient(emotionDict) {
  const entries = Object.entries(emotionDict)
    .filter(([, v]) => v > 0.03)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
  if (!entries.length) return null;
  if (entries.length === 1) {
    const rgb = EmotionPalette[entries[0][0]] || EmotionPalette.default;
    return `rgba(${rgb}, 0.14)`;
  }
  const stops = entries.map(([emo], i) => {
    const rgb = EmotionPalette[emo] || EmotionPalette.default;
    const pct = Math.round((i / (entries.length - 1)) * 100);
    return `rgba(${rgb}, 0.15) ${pct}%`;
  });
  return `linear-gradient(135deg, ${stops.join(', ')})`;
}

// ── Demo messages ─────────────────────────────────────────────────────────────

function buildDemoMessages(myName, otherName) {
  return [
    {
      id: 'demo-1', sender: 'ai', senderName: otherName,
      text: "Sure, totally fine with that 🙄",
      analysis: {
        type: 'analysis',
        data: {
          id: 'demo-1', raw_text: "Sure, totally fine with that 🙄",
          final_dominant_emotion: 'annoyance', final_valence: 0.42,
          meta_confidence: 0.81,
          bert_emotions: [
            { label: 'annoyance',   score: 0.48 },
            { label: 'disgust',     score: 0.22 },
            { label: 'disapproval', score: 0.15 },
            { label: 'neutral',     score: 0.10 },
            { label: 'joy',         score: 0.05 },
          ],
          logic_map: { VADER: 0.12, BERT: 0.28, GoEmotions: 0.53, Context: 0.07 },
          sender_id: 'demo-other',
        },
        ai_insight: 'Sarcasm detected: positive lexical content paired with dismissive 🙄. Sarcasm score: 0.90.',
      },
    },
    {
      id: 'demo-2', sender: 'user', senderName: myName,
      text: "I'm genuinely loving this!! Everything is perfect 🔥",
      analysis: {
        type: 'analysis',
        data: {
          id: 'demo-2', raw_text: "I'm genuinely loving this!! Everything is perfect 🔥",
          final_dominant_emotion: 'joy', final_valence: 0.94,
          meta_confidence: 0.93,
          bert_emotions: [
            { label: 'joy',        score: 0.62 },
            { label: 'excitement', score: 0.24 },
            { label: 'love',       score: 0.08 },
            { label: 'admiration', score: 0.04 },
            { label: 'neutral',    score: 0.02 },
          ],
          logic_map: { VADER: 0.38, BERT: 0.22, GoEmotions: 0.36, Context: 0.04 },
          sender_id: 'demo-me',
        },
      },
    },
    {
      id: 'demo-3', sender: 'user', senderName: myName,
      text: "Actually… I don't know anymore. This whole thing is overwhelming me.",
      analysis: {
        type: 'analysis',
        data: {
          id: 'demo-3', raw_text: "Actually… I don't know anymore. This whole thing is overwhelming me.",
          final_dominant_emotion: 'sadness', final_valence: -0.68,
          meta_confidence: 0.79,
          bert_emotions: [
            { label: 'sadness',        score: 0.45 },
            { label: 'fear',           score: 0.22 },
            { label: 'disappointment', score: 0.18 },
            { label: 'nervousness',    score: 0.10 },
            { label: 'confusion',      score: 0.05 },
          ],
          logic_map: { VADER: 0.30, BERT: 0.32, GoEmotions: 0.28, Context: 0.10 },
          context_shift: { type: 'Context Shift', from: 'joy', to: 'sadness', significance: 'High' },
          sender_id: 'demo-me',
        },
        ai_insight: 'Sharp valence cliff (Δ = −1.62 in 1 message). Velocity: FAST. Possible emotional withdrawal. Arc: DESCENDING.',
      },
    },
    {
      id: 'demo-4', sender: 'ai', senderName: otherName,
      text: "Hey, I hear you. I actually do care about how you're doing, despite everything.",
      analysis: {
        type: 'analysis',
        data: {
          id: 'demo-4', raw_text: "Hey, I hear you. I actually do care about how you're doing, despite everything.",
          final_dominant_emotion: 'caring', final_valence: 0.58,
          meta_confidence: 0.85,
          bert_emotions: [
            { label: 'caring',    score: 0.44 },
            { label: 'love',      score: 0.21 },
            { label: 'gratitude', score: 0.16 },
            { label: 'remorse',   score: 0.12 },
            { label: 'relief',    score: 0.07 },
          ],
          logic_map: { VADER: 0.20, BERT: 0.35, GoEmotions: 0.35, Context: 0.10 },
          sender_id: 'demo-other',
        },
      },
    },
    {
      id: 'demo-5', sender: 'user', senderName: myName,
      text: "Alright. Let's do this together. I think we can actually make it work.",
      analysis: {
        type: 'analysis',
        data: {
          id: 'demo-5', raw_text: "Alright. Let's do this together. I think we can actually make it work.",
          final_dominant_emotion: 'optimism', final_valence: 0.72,
          meta_confidence: 0.87,
          bert_emotions: [
            { label: 'optimism',  score: 0.38 },
            { label: 'relief',    score: 0.28 },
            { label: 'approval',  score: 0.18 },
            { label: 'gratitude', score: 0.10 },
            { label: 'joy',       score: 0.06 },
          ],
          logic_map: { VADER: 0.25, BERT: 0.28, GoEmotions: 0.37, Context: 0.10 },
          sender_id: 'demo-me',
        },
        ai_insight: 'Arc recovery: RAGS_TO_RICHES pattern. Valence slope: +2.1 over 3 messages. Trajectory: ASCENDING.',
      },
    },
  ];
}

// ── Avatar ────────────────────────────────────────────────────────────────────

function Avatar({ name = '?', size = 44, rgb = '88,86,214', online = false }) {
  const letter = name?.[0]?.toUpperCase() || '?';
  return (
    <div style={{ position: 'relative', flexShrink: 0 }}>
      <div style={{
        width: size, height: size, borderRadius: '50%',
        background: `linear-gradient(135deg, rgba(${rgb},.25), rgba(${rgb},.12))`,
        border: `2px solid rgba(${rgb},.40)`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: size * 0.40, fontWeight: 700, color: `rgb(${rgb})`,
        boxShadow: `0 0 0 2px rgba(${rgb},.12)`,
      }}>
        {letter}
      </div>
      {online && (
        <div style={{
          position: 'absolute', bottom: 1, right: 1,
          width: size * 0.28, height: size * 0.28, borderRadius: '50%',
          background: '#22c55e', border: '2px solid #ffffff',
          boxShadow: '0 0 5px rgba(34,197,94,.6)',
        }} />
      )}
    </div>
  );
}

// ── MsgBubble ─────────────────────────────────────────────────────────────────

const PIPELINE_MODELS = [
  { key: 'vader',       label: 'VADER' },
  { key: 'basic_bert',  label: 'BERT'  },
  { key: 'go_emotions', label: 'GoE'   },
];

function MsgBubble({ msg, isOwn, onClick, isRegenerating, onDelete, partialModels }) {
  const [hovered, setHovered] = useState(false);

  const bert = msg.analysis?.data?.bert_emotions;
  let emotionDict = {};
  if (bert?.length) bert.forEach(({ label, score }) => { emotionDict[label] = score; });
  const hasAnalysis = Object.keys(emotionDict).length > 0;

  const dom    = msg.analysis?.data?.final_dominant_emotion;
  const domRgb = dom ? emotionRgb(dom) : null;

  const ownBg    = hasAnalysis ? (bubbleGradient(emotionDict) || 'rgba(0,119,255,0.10)') : 'rgba(0,119,255,0.10)';
  const borderClr = isOwn
    ? (domRgb ? `rgba(${domRgb},.35)` : 'rgba(0,119,255,.25)')
    : 'rgba(0,0,0,.07)';

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => msg.analysis && onClick?.(msg)}
      style={{
        display: 'flex', flexDirection: 'column',
        alignItems: isOwn ? 'flex-end' : 'flex-start',
        marginBottom: 3,
        cursor: msg.analysis ? 'pointer' : 'default',
        position: 'relative',
      }}
    >
      <div style={{
        maxWidth: '70%',
        padding: '10px 14px',
        borderRadius: isOwn ? '22px 22px 6px 22px' : '22px 22px 22px 6px',
        background: isOwn ? ownBg : '#f0f0f0',
        border: `1.5px solid ${borderClr}`,
        fontSize: '0.94rem', lineHeight: 1.5,
        color: '#1c1c2e',
        wordBreak: 'break-word',
        boxShadow: domRgb && isOwn
          ? `0 2px 12px rgba(${domRgb},.14), inset 0 1px 0 rgba(255,255,255,.75)`
          : isOwn
            ? '0 2px 12px rgba(0,119,255,.09), inset 0 1px 0 rgba(255,255,255,.75)'
            : '0 1px 3px rgba(0,0,0,.07)',
        backdropFilter: isOwn ? 'blur(20px) saturate(180%)' : 'none',
        WebkitBackdropFilter: isOwn ? 'blur(20px) saturate(180%)' : 'none',
        animation: 'igMsgIn .26s cubic-bezier(.34,1.2,.64,1) both',
        position: 'relative', overflow: 'hidden',
        outline: hovered && msg.analysis ? `2px solid rgba(${domRgb || '0,119,255'},.30)` : 'none',
        outlineOffset: 1,
        transition: 'outline-color .15s',
      }}>
        {isOwn && (
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, height: '38%',
            background: 'linear-gradient(180deg,rgba(255,255,255,.50) 0%,transparent 100%)',
            pointerEvents: 'none',
          }} />
        )}

        {isRegenerating ? (
          <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#9ca3af', fontStyle: 'italic', fontSize: '0.87rem' }}>
            <span className="regen-spinner" style={{
              width: 12, height: 12, border: '2px solid rgba(0,0,0,.10)',
              borderTopColor: '#6b7280', borderRadius: '50%',
              display: 'inline-block', animation: 'regen-spin .7s linear infinite',
            }} />
            Re-analyzing…
          </span>
        ) : msg.text}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 3 }}>
        {dom && dom.toLowerCase() !== 'neutral' && (
          <div style={{
            fontSize: '0.66rem', color: domRgb ? `rgb(${domRgb})` : '#9ca3af',
            fontWeight: 600, letterSpacing: '.04em', opacity: .75,
            textTransform: 'capitalize',
          }}>
            {dom}
          </div>
        )}
        {msg.analysis && hovered && (
          <div style={{ fontSize: '0.62rem', color: '#9ca3af' }}>click to inspect ↗</div>
        )}
        {!msg.analysis && msg.sender === 'user' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            {PIPELINE_MODELS.map(({ key, label }) => {
              const done = partialModels?.has(key);
              return (
                <div key={key} style={{
                  fontSize: '0.60rem', fontWeight: 700,
                  padding: '1px 5px', borderRadius: 99,
                  background: done ? 'rgba(0,119,255,0.13)' : 'rgba(0,0,0,0.05)',
                  color: done ? '#0077ff' : '#b0b7c3',
                  border: `1px solid ${done ? 'rgba(0,119,255,0.30)' : 'rgba(0,0,0,0.08)'}`,
                  transition: 'all 0.25s ease',
                }}>
                  {done ? '✓ ' : ''}{label}
                </div>
              );
            })}
            <span style={{ fontSize: '0.64rem', color: '#9ca3af', marginLeft: 2 }}>
              {partialModels?.size === 3 ? 'Finalizing…' : 'Processing…'}
            </span>
          </div>
        )}
        {isOwn && hovered && onDelete && (
          <button
            onClick={e => { e.stopPropagation(); onDelete(msg.id); }}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              fontSize: '0.65rem', color: '#ef4444', padding: '0 2px',
              lineHeight: 1, opacity: 0.7,
            }}
            title="Delete message"
          >✕</button>
        )}
      </div>
    </div>
  );
}

// ── Group Modal ───────────────────────────────────────────────────────────────

function GroupModal({ globalUsers, currentUser, onConfirm, onClose }) {
  const [groupName, setGroupName]     = useState('');
  const [selected, setSelected]       = useState([]);
  const [search, setSearch]           = useState('');
  const [error, setError]             = useState('');
  const [submitting, setSubmitting]   = useState(false);

  const eligible = globalUsers.filter(u =>
    u.user_id !== currentUser?.user_id &&
    u.display_name?.toLowerCase().includes(search.toLowerCase())
  );

  const toggle = (uid) =>
    setSelected(prev => prev.includes(uid) ? prev.filter(x => x !== uid) : [...prev, uid]);

  const handleSubmit = async () => {
    if (!groupName.trim()) { setError('Group name is required.'); return; }
    if (selected.length === 0) { setError('Add at least one member.'); return; }
    setSubmitting(true);
    try {
      await onConfirm(groupName.trim(), selected);
      onClose();
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Failed to create group.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(0,0,0,.50)', display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(8px)' }}
      onClick={onClose}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{ background: '#fff', borderRadius: 20, width: 420, maxHeight: '80vh', display: 'flex', flexDirection: 'column', boxShadow: '0 20px 60px rgba(0,0,0,.20)', overflow: 'hidden', animation: 'igModalIn .22s cubic-bezier(.34,1.2,.64,1) both' }}
      >
        <div style={{ padding: '20px 20px 0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <button onClick={onClose} style={{ background: 'none', border: 'none', fontSize: '1.4rem', cursor: 'pointer', color: '#1c1c2e', lineHeight: 1 }}>✕</button>
          <span style={{ fontWeight: 700, fontSize: '1rem', color: '#1c1c2e' }}>New Group</span>
          <div style={{ width: 24 }} />
        </div>

        <div style={{ padding: '14px 20px 0' }}>
          <input
            autoFocus
            placeholder="Group name…"
            value={groupName}
            onChange={e => setGroupName(e.target.value)}
            style={{ width: '100%', boxSizing: 'border-box', background: '#f3f4f6', border: 'none', borderRadius: 12, padding: '10px 14px', fontSize: '0.93rem', color: '#1c1c2e', outline: 'none' }}
          />
        </div>

        <div style={{ padding: '10px 20px 4px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#f3f4f6', borderRadius: 12, padding: '10px 14px' }}>
            <svg width="15" height="15" fill="none" stroke="#9ca3af" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <input
              placeholder="Search people…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{ background: 'none', border: 'none', outline: 'none', fontSize: '0.92rem', color: '#1c1c2e', flex: 1 }}
            />
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '6px 0 8px' }}>
          {eligible.map(u => (
            <button key={u.user_id} onClick={() => toggle(u.user_id)}
              style={{ width: '100%', background: selected.includes(u.user_id) ? '#f0f6ff' : 'none', border: 'none', display: 'flex', alignItems: 'center', gap: 12, padding: '10px 20px', cursor: 'pointer', textAlign: 'left', transition: 'background .12s' }}
              onMouseOver={e => { if (!selected.includes(u.user_id)) e.currentTarget.style.background = '#f8f9fa'; }}
              onMouseOut={e => { if (!selected.includes(u.user_id)) e.currentTarget.style.background = 'none'; }}
            >
              <Avatar name={u.display_name} size={36} rgb="0,119,255" />
              <span style={{ flex: 1, fontWeight: 500, fontSize: '0.93rem', color: '#1c1c2e' }}>{u.display_name}</span>
              <div style={{ width: 20, height: 20, borderRadius: '50%', border: `2px solid ${selected.includes(u.user_id) ? '#0077ff' : '#d1d5db'}`, background: selected.includes(u.user_id) ? '#0077ff' : 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.65rem', color: '#fff', transition: 'all .15s' }}>
                {selected.includes(u.user_id) ? '✓' : ''}
              </div>
            </button>
          ))}
          {eligible.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: '#9ca3af', fontSize: '0.88rem' }}>No users found.</div>
          )}
        </div>

        {error && <div style={{ padding: '0 20px 8px', fontSize: '0.82rem', color: '#ef4444' }}>{error}</div>}

        <div style={{ padding: '12px 20px 20px', borderTop: '1px solid #f0f0f0', display: 'flex', gap: 10 }}>
          <button onClick={onClose} style={{ flex: 1, background: '#f3f4f6', color: '#374151', border: 'none', borderRadius: 12, padding: '11px', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem' }}>
            Cancel
          </button>
          <button onClick={handleSubmit} disabled={submitting}
            style={{ flex: 1, background: '#0077ff', color: '#fff', border: 'none', borderRadius: 12, padding: '11px', cursor: submitting ? 'wait' : 'pointer', fontWeight: 700, fontSize: '0.88rem', opacity: submitting ? 0.7 : 1 }}>
            {submitting ? 'Creating…' : `Create (${selected.length} members)`}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function IGDashboard({
  currentUser,
  conversations,
  globalUsers,
  onlineUsers = new Set(),
  activeConversationId,
  onSelectConversation,
  onCreateChat,
  onCreateGroup,
  onAddMember,
  onRemoveMember,
  onDeleteMessage,
  onGoToAnalytics,
  onGoToLiveAnalytics,
  onGoToAdmin,
  onLogout,
  status,
  messages,
  inputValue,
  setInputValue,
  onSend,
  currentAnalysis,
  processing = false,
  partialModels = new Set(),
  onRegenerateAnalysis,
  onInjectDemo,
  regeneratingIds = new Set(),
  socketRef,
  onDemoStart,
}) {
  const [search, setSearch]                   = useState('');
  const [showCompose, setShowCompose]         = useState(false);
  const [showGroupModal, setShowGroupModal]   = useState(false);
  const [composeSearch, setComposeSearch]     = useState('');
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [selectedMsg, setSelectedMsg]         = useState(null);
  const [memberPanelOpen, setMemberPanelOpen] = useState(false);
  const [memberError, setMemberError]         = useState('');
  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'auto' });
  }, [messages]);

  useEffect(() => {
    if (activeConversationId) inputRef.current?.focus();
  }, [activeConversationId]);

  useEffect(() => { setSelectedMsg(null); setMemberPanelOpen(false); setMemberError(''); }, [activeConversationId]);

  // Conversation filtering — works for both direct (other_display_name) and group (name)
  const filteredConvs = conversations.filter(c => {
    const label = c.type === 'group' ? c.name : c.other_display_name;
    return label?.toLowerCase().includes(search.toLowerCase());
  });

  const filteredUsers = globalUsers.filter(u =>
    u.display_name?.toLowerCase().includes(composeSearch.toLowerCase())
  );

  const activeConv  = conversations.find(c => c.conversation_id === activeConversationId);
  const isGroup     = activeConv?.type === 'group';
  const isCreator   = isGroup && activeConv?.creator_user_id === currentUser?.user_id;
  const activeRgb   = emotionRgb(activeConv?.dominant_emotion);
  const dominant    = currentAnalysis?.data?.final_dominant_emotion;
  const dominantRgb = dominant ? emotionRgb(dominant) : null;

  // For groups: label + avatar use group name; for direct: use other_display_name
  const convLabel = (conv) => conv.type === 'group' ? conv.name : conv.other_display_name;
  const activeLabel = activeConv ? convLabel(activeConv) : '';

  const rightPanelOpen = !!activeConversationId;

  const handleAddMember = async (userId) => {
    setMemberError('');
    try {
      await onAddMember(activeConversationId, userId);
    } catch (e) {
      setMemberError(e?.response?.data?.detail || e.message || 'Failed to add member.');
    }
  };

  const handleRemoveMember = async (userId) => {
    setMemberError('');
    try {
      await onRemoveMember(activeConversationId, userId);
    } catch (e) {
      setMemberError(e?.response?.data?.detail || e.message || 'Failed to remove member.');
    }
  };

  return (
    <div style={{
      display: 'flex', height: '100vh',
      background: '#ffffff',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      position: 'relative',
    }}>

      {/* ── Compose Modal ─────────────────────────────────────────────── */}
      {showCompose && (
        <div
          style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(0,0,0,.50)', display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(8px)' }}
          onClick={() => { setShowCompose(false); setComposeSearch(''); }}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{ background: '#fff', borderRadius: 20, width: 400, maxHeight: '70vh', display: 'flex', flexDirection: 'column', boxShadow: '0 20px 60px rgba(0,0,0,.20)', overflow: 'hidden', animation: 'igModalIn .22s cubic-bezier(.34,1.2,.64,1) both' }}
          >
            <div style={{ padding: '20px 20px 0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <button onClick={() => { setShowCompose(false); setComposeSearch(''); }} style={{ background: 'none', border: 'none', fontSize: '1.4rem', cursor: 'pointer', color: '#1c1c2e', lineHeight: 1 }}>✕</button>
              <span style={{ fontWeight: 700, fontSize: '1rem', color: '#1c1c2e' }}>New Message</span>
              <div style={{ width: 24 }} />
            </div>

            <div style={{ padding: '14px 20px 0' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#f3f4f6', borderRadius: 12, padding: '10px 14px' }}>
                <svg width="16" height="16" fill="none" stroke="#9ca3af" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                <input
                  autoFocus placeholder="Search people…"
                  value={composeSearch} onChange={e => setComposeSearch(e.target.value)}
                  style={{ background: 'none', border: 'none', outline: 'none', fontSize: '0.93rem', color: '#1c1c2e', flex: 1 }}
                />
              </div>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: '10px 0 8px' }}>
              {filteredUsers.length === 0 && (
                <div style={{ padding: '20px', textAlign: 'center', color: '#9ca3af', fontSize: '0.88rem' }}>No users found.</div>
              )}
              {filteredUsers.map(u => {
                const isOnline = onlineUsers.has(u.user_id);
                return (
                  <button key={u.user_id} onClick={() => { onCreateChat(u.user_id); setShowCompose(false); setComposeSearch(''); }}
                    style={{ width: '100%', background: 'none', border: 'none', display: 'flex', alignItems: 'center', gap: 12, padding: '10px 20px', cursor: 'pointer', transition: 'background .12s', textAlign: 'left' }}
                    onMouseOver={e => e.currentTarget.style.background = '#f8f9fa'}
                    onMouseOut={e => e.currentTarget.style.background = 'none'}
                  >
                    <Avatar name={u.display_name} size={44} rgb="0,119,255" online={isOnline} />
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '0.93rem', color: '#1c1c2e' }}>{u.display_name}</div>
                      <div style={{ fontSize: '0.78rem', color: isOnline ? '#22c55e' : '#9ca3af' }}>
                        {isOnline ? 'Active now' : 'Offline'}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── Group Creation Modal ───────────────────────────────────────── */}
      {showGroupModal && (
        <GroupModal
          globalUsers={globalUsers}
          currentUser={currentUser}
          onConfirm={onCreateGroup}
          onClose={() => setShowGroupModal(false)}
        />
      )}

      {/* ── Left Sidebar ──────────────────────────────────────────────── */}
      <div style={{ width: 360, borderRight: '1px solid #efefef', display: 'flex', flexDirection: 'column', background: '#ffffff', flexShrink: 0 }}>
        {/* Header */}
        <div style={{ padding: '20px 20px 12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div
            style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', position: 'relative' }}
            onClick={() => setShowProfileMenu(p => !p)}
          >
            <span style={{ fontWeight: 800, fontSize: '1.05rem', color: '#1c1c2e' }}>{currentUser?.display_name}</span>
            <svg width="14" height="14" fill="none" stroke="#1c1c2e" strokeWidth="2.5" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>

            {showProfileMenu && (
              <div
                style={{ position: 'absolute', top: '100%', left: 0, zIndex: 100, background: '#fff', borderRadius: 14, minWidth: 200, boxShadow: '0 8px 32px rgba(0,0,0,.14)', border: '1px solid rgba(0,0,0,.07)', overflow: 'hidden', marginTop: 8, animation: 'igModalIn .18s ease both' }}
                onClick={e => e.stopPropagation()}
              >
                <div style={{ padding: '14px 18px', borderBottom: '1px solid #f0f0f0' }}>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem', color: '#1c1c2e' }}>{currentUser?.display_name}</div>
                  <div style={{ fontSize: '0.78rem', color: '#9ca3af', marginTop: 2 }}>{currentUser?.email}</div>
                  {currentUser?.role === 'admin' && (
                    <span style={{ display: 'inline-block', marginTop: 4, background: 'linear-gradient(135deg,#f59e0b,#d97706)', color: '#fff', fontSize: '0.65rem', fontWeight: 700, padding: '2px 8px', borderRadius: 6, letterSpacing: '.06em' }}>ADMIN</span>
                  )}
                </div>
                <button onClick={() => { setShowProfileMenu(false); onGoToAnalytics(); }}
                  style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: '11px 18px', textAlign: 'left', fontSize: '0.9rem', color: '#1c1c2e', display: 'flex', alignItems: 'center', gap: 10 }}
                  onMouseOver={e => e.currentTarget.style.background = '#f8f9fa'}
                  onMouseOut={e => e.currentTarget.style.background = 'none'}
                >
                  <span>✨</span> Analytics
                </button>
                {onGoToLiveAnalytics && (
                  <button onClick={() => { setShowProfileMenu(false); onGoToLiveAnalytics(); }}
                    style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: '11px 18px', textAlign: 'left', fontSize: '0.9rem', color: '#1c1c2e', display: 'flex', alignItems: 'center', gap: 10 }}
                    onMouseOver={e => e.currentTarget.style.background = '#f8f9fa'}
                    onMouseOut={e => e.currentTarget.style.background = 'none'}
                  >
                    <span>📊</span> Live Analytics
                  </button>
                )}
                {currentUser?.role === 'admin' && (
                  <button onClick={() => { setShowProfileMenu(false); onGoToAdmin(); }}
                    style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: '11px 18px', textAlign: 'left', fontSize: '0.9rem', color: '#7c3aed', display: 'flex', alignItems: 'center', gap: 10 }}
                    onMouseOver={e => e.currentTarget.style.background = '#f5f3ff'}
                    onMouseOut={e => e.currentTarget.style.background = 'none'}
                  >
                    <span>🔬</span> Pipeline Inspector
                  </button>
                )}
                <div style={{ borderTop: '1px solid #f0f0f0' }}>
                  <button onClick={onLogout}
                    style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: '11px 18px', textAlign: 'left', fontSize: '0.9rem', color: '#ef4444', display: 'flex', alignItems: 'center', gap: 10 }}
                    onMouseOver={e => e.currentTarget.style.background = '#fff5f5'}
                    onMouseOut={e => e.currentTarget.style.background = 'none'}
                  >
                    <span>→</span> Log out
                  </button>
                </div>
              </div>
            )}
          </div>

          <button onClick={() => { setShowCompose(true); setShowProfileMenu(false); }}
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 8, borderRadius: 10, transition: 'background .12s', display: 'flex', alignItems: 'center' }}
            title="New Message"
            onMouseOver={e => e.currentTarget.style.background = '#f3f4f6'}
            onMouseOut={e => e.currentTarget.style.background = 'none'}
          >
            <svg width="22" height="22" fill="none" stroke="#1c1c2e" strokeWidth="2" viewBox="0 0 24 24">
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
            </svg>
          </button>
        </div>

        {/* Search */}
        <div style={{ padding: '0 16px 12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#f3f4f6', borderRadius: 12, padding: '10px 14px' }}>
            <svg width="15" height="15" fill="none" stroke="#9ca3af" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <input placeholder="Search…" value={search} onChange={e => setSearch(e.target.value)}
              style={{ background: 'none', border: 'none', outline: 'none', fontSize: '0.92rem', color: '#1c1c2e', flex: 1 }}
            />
          </div>
        </div>

        {/* Section label + WS status */}
        <div style={{ padding: '0 20px 10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontWeight: 700, fontSize: '0.88rem', color: '#1c1c2e' }}>Messages</span>
          <span style={{ fontSize: '0.78rem', color: status === 'Live' ? '#22c55e' : '#9ca3af', display: 'flex', alignItems: 'center', gap: 4, fontWeight: 500 }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: status === 'Live' ? '#22c55e' : '#d1d5db', display: 'inline-block', boxShadow: status === 'Live' ? '0 0 5px #22c55e' : 'none' }} />
            {status}
          </span>
        </div>

        {/* Conversation list */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {filteredConvs.length === 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '40px 20px', gap: 12, color: '#9ca3af' }}>
              <div style={{ width: 56, height: 56, borderRadius: '50%', background: '#f3f4f6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.5rem' }}>💬</div>
              <p style={{ margin: 0, fontSize: '0.9rem', textAlign: 'center' }}>No conversations yet.</p>
              <button onClick={() => setShowCompose(true)} style={{ background: '#0077ff', color: '#fff', border: 'none', borderRadius: 10, padding: '10px 20px', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem' }}>
                Send a message
              </button>
            </div>
          )}

          {filteredConvs.map(conv => {
            const rgb      = emotionRgb(conv.dominant_emotion);
            const isActive = conv.conversation_id === activeConversationId;
            const label    = convLabel(conv);
            const lastMsg  = messages.length > 0 && isActive
              ? messages[messages.length - 1]?.text
              : conv.dominant_emotion || 'Start chatting';

            return (
              <button key={conv.conversation_id}
                onClick={() => { onSelectConversation(conv.conversation_id); setShowProfileMenu(false); }}
                style={{
                  width: '100%', background: isActive ? '#f0f6ff' : 'none',
                  border: 'none', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 16px', transition: 'background .12s', textAlign: 'left',
                  borderLeft: isActive ? '3px solid #0077ff' : '3px solid transparent',
                }}
                onMouseOver={e => { if (!isActive) e.currentTarget.style.background = '#f8f9fa'; }}
                onMouseOut={e => { if (!isActive) e.currentTarget.style.background = 'none'; }}
              >
                <Avatar name={label} size={52} rgb={rgb} online={conv.type !== 'group' && onlineUsers.has(conv.other_user_id)} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ fontWeight: isActive ? 700 : 600, fontSize: '0.93rem', color: '#1c1c2e' }}>{label}</span>
                      {conv.type === 'group' && (
                        <span style={{ fontSize: '0.65rem', background: '#f0f6ff', color: '#0077ff', fontWeight: 700, padding: '1px 6px', borderRadius: 6, letterSpacing: '.04em' }}>
                          GROUP · {conv.member_count || 0}
                        </span>
                      )}
                    </div>
                    <span style={{ fontSize: '0.72rem', color: '#9ca3af', flexShrink: 0 }}>{timeAgo(conv.last_message_time || conv.conversation_id)}</span>
                  </div>
                  <div style={{ fontSize: '0.82rem', color: '#6b7280', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: 2 }}>
                    {typeof lastMsg === 'string' ? lastMsg : '…'}
                  </div>
                  {conv.dominant_emotion && (
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, marginTop: 3, fontSize: '0.68rem', color: `rgb(${rgb})`, fontWeight: 600, letterSpacing: '.04em', textTransform: 'capitalize' }}>
                      <div style={{ width: 6, height: 6, borderRadius: '50%', background: `rgb(${rgb})`, boxShadow: `0 0 4px rgba(${rgb},.5)` }} />
                      {conv.dominant_emotion}
                    </div>
                  )}
                </div>
              </button>
            );
          })}
        </div>

        {/* Footer */}
        <div style={{ padding: '12px 16px', borderTop: '1px solid #f0f0f0', display: 'flex', gap: 6 }}>
          <button onClick={() => setShowCompose(true)} style={{
            flex: 1, background: '#0077ff', color: '#fff', border: 'none',
            borderRadius: 12, padding: '11px', cursor: 'pointer', fontWeight: 700, fontSize: '0.88rem',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
            boxShadow: '0 4px 16px rgba(0,119,255,.25)', transition: 'opacity .15s, transform .12s',
          }}
            onMouseOver={e => { e.currentTarget.style.opacity = '.9'; e.currentTarget.style.transform = 'scale(1.01)'; }}
            onMouseOut={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.transform = 'scale(1)'; }}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            New Message
          </button>
          <button onClick={() => setShowGroupModal(true)}
            style={{ background: '#f3f4f6', color: '#374151', border: 'none', borderRadius: 12, padding: '11px 14px', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem', transition: 'background .12s' }}
            title="New Group Chat"
            onMouseOver={e => e.currentTarget.style.background = '#e5e7eb'}
            onMouseOut={e => e.currentTarget.style.background = '#f3f4f6'}
          >
            👥
          </button>
          <button onClick={onGoToAnalytics}
            style={{ background: '#f3f4f6', color: '#374151', border: 'none', borderRadius: 12, padding: '11px 14px', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem', transition: 'background .12s' }}
            title="View Insights"
            onMouseOver={e => e.currentTarget.style.background = '#e5e7eb'}
            onMouseOut={e => e.currentTarget.style.background = '#f3f4f6'}
          >
            ◎
          </button>
        </div>
      </div>

      {/* ── Chat Area ─────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {!activeConversationId ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16 }}>
            <div style={{ width: 96, height: 96, borderRadius: '50%', border: '3px solid #1c1c2e', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '2.6rem' }}>💬</div>
            <div style={{ textAlign: 'center' }}>
              <h2 style={{ margin: '0 0 6px', fontWeight: 700, fontSize: '1.2rem', color: '#1c1c2e' }}>Your Messages</h2>
              <p style={{ margin: 0, color: '#6b7280', fontSize: '0.9rem' }}>Emotion-aware private messaging — select a conversation to begin.</p>
            </div>
            <button onClick={() => setShowCompose(true)} style={{ background: '#0077ff', color: '#fff', border: 'none', borderRadius: 12, padding: '11px 28px', cursor: 'pointer', fontWeight: 700, fontSize: '0.93rem', boxShadow: '0 4px 16px rgba(0,119,255,.25)' }}>
              Send message
            </button>
          </div>
        ) : (
          <>
            {/* Chat header */}
            <div style={{ padding: '14px 20px', borderBottom: '1px solid #efefef', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: '#ffffff', position: 'sticky', top: 0, zIndex: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <Avatar name={activeLabel} size={42} rgb={activeRgb} online={!isGroup && onlineUsers.has(activeConv?.other_user_id)} />
                <div>
                  <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#1c1c2e', display: 'flex', alignItems: 'center', gap: 6 }}>
                    {activeLabel}
                    {isGroup && (
                      <span style={{ fontSize: '0.65rem', background: '#f0f6ff', color: '#0077ff', fontWeight: 700, padding: '1px 6px', borderRadius: 6 }}>
                        GROUP · {activeConv?.member_count || 0}
                      </span>
                    )}
                  </div>
                  {dominant ? (
                    <div style={{ fontSize: '0.73rem', fontWeight: 600, color: dominantRgb ? `rgb(${dominantRgb})` : '#22c55e', display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 7, height: 7, borderRadius: '50%', background: dominantRgb ? `rgb(${dominantRgb})` : '#22c55e', display: 'inline-block' }} />
                      {dominant}
                    </div>
                  ) : !isGroup && onlineUsers.has(activeConv?.other_user_id) ? (
                    <div style={{ fontSize: '0.73rem', color: '#22c55e' }}>Active now</div>
                  ) : (
                    <div style={{ fontSize: '0.73rem', color: '#9ca3af' }}>{isGroup ? 'Group chat' : 'Offline'}</div>
                  )}
                </div>
              </div>

              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                {isGroup && (
                  <button onClick={() => setMemberPanelOpen(p => !p)}
                    title="Manage members"
                    style={{ background: memberPanelOpen ? '#f0f6ff' : 'none', border: `1px solid ${memberPanelOpen ? '#0077ff' : 'rgba(0,0,0,.10)'}`, borderRadius: 8, padding: '5px 10px', cursor: 'pointer', fontSize: '0.75rem', color: memberPanelOpen ? '#0077ff' : '#6b7280', fontWeight: 500, transition: 'all .15s' }}>
                    👥 Members
                  </button>
                )}

                {socketRef && onDemoStart && (
                  <DemoRunner
                    currentUser={currentUser}
                    socketRef={socketRef}
                    onDemoStart={onDemoStart}
                  />
                )}

                {onInjectDemo && (
                  <button
                    onClick={() => {
                      const demos = buildDemoMessages(currentUser.display_name, activeLabel || 'Other');
                      onInjectDemo(demos);
                    }}
                    title="Inject demo conversation with pre-built ML analysis"
                    style={{ background: 'rgba(0,0,0,.04)', border: '1px solid rgba(0,0,0,.08)', borderRadius: 8, padding: '5px 10px', cursor: 'pointer', fontSize: '0.70rem', color: '#9ca3af', fontWeight: 500, transition: 'background .12s, color .12s' }}
                    onMouseOver={e => { e.currentTarget.style.background = '#f3f4f6'; e.currentTarget.style.color = '#374151'; }}
                    onMouseOut={e => { e.currentTarget.style.background = 'rgba(0,0,0,.04)'; e.currentTarget.style.color = '#9ca3af'; }}
                  >
                    ⚗ Demo
                  </button>
                )}

                <button onClick={onGoToAnalytics} title="View Insights"
                  style={{ background: 'none', border: 'none', borderRadius: 10, width: 38, height: 38, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#1c1c2e', transition: 'background .12s' }}
                  onMouseOver={e => e.currentTarget.style.background = '#f3f4f6'}
                  onMouseOut={e => e.currentTarget.style.background = 'none'}
                >
                  <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                </button>
              </div>
            </div>

            {/* Member panel for groups */}
            {isGroup && memberPanelOpen && (
              <div style={{ borderBottom: '1px solid #efefef', padding: '12px 20px', background: '#fafafa', maxHeight: 220, overflowY: 'auto' }}>
                <div style={{ fontWeight: 600, fontSize: '0.82rem', color: '#6b7280', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>
                  Members ({activeConv?.member_count || 0})
                </div>
                {(activeConv?.members || []).map(m => {
                  const isMe      = m.user_id === currentUser?.user_id;
                  const isOwner   = m.user_id === activeConv?.creator_user_id;
                  return (
                    <div key={m.user_id} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                      <Avatar name={m.display_name} size={28} rgb="0,119,255" online={onlineUsers.has(m.user_id)} />
                      <span style={{ flex: 1, fontSize: '0.88rem', color: '#1c1c2e' }}>{m.display_name}{isMe ? ' (you)' : ''}</span>
                      {isOwner && <span style={{ fontSize: '0.65rem', background: '#fef3c7', color: '#d97706', padding: '1px 6px', borderRadius: 6, fontWeight: 700 }}>★ Creator</span>}
                      {isCreator && !isMe && (
                        <button onClick={() => handleRemoveMember(m.user_id)}
                          style={{ background: 'none', border: '1px solid #fca5a5', borderRadius: 6, padding: '2px 8px', cursor: 'pointer', fontSize: '0.72rem', color: '#ef4444', transition: 'background .12s' }}
                          onMouseOver={e => e.currentTarget.style.background = '#fff5f5'}
                          onMouseOut={e => e.currentTarget.style.background = 'none'}
                        >Remove</button>
                      )}
                    </div>
                  );
                })}

                {isCreator && (
                  <div style={{ marginTop: 10, borderTop: '1px solid #efefef', paddingTop: 10 }}>
                    <div style={{ fontWeight: 600, fontSize: '0.78rem', color: '#9ca3af', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.05em' }}>Add member</div>
                    {globalUsers
                      .filter(u => !(activeConv?.members || []).some(m => m.user_id === u.user_id))
                      .map(u => (
                        <button key={u.user_id} onClick={() => handleAddMember(u.user_id)}
                          style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', background: 'none', border: 'none', padding: '5px 0', cursor: 'pointer', textAlign: 'left' }}
                          onMouseOver={e => e.currentTarget.style.opacity = '0.7'}
                          onMouseOut={e => e.currentTarget.style.opacity = '1'}
                        >
                          <Avatar name={u.display_name} size={22} rgb="0,119,255" />
                          <span style={{ fontSize: '0.85rem', color: '#374151' }}>{u.display_name}</span>
                          <span style={{ fontSize: '0.72rem', color: '#0077ff', marginLeft: 'auto', fontWeight: 600 }}>+ Add</span>
                        </button>
                      ))
                    }
                  </div>
                )}
                {memberError && <div style={{ fontSize: '0.78rem', color: '#ef4444', marginTop: 6 }}>{memberError}</div>}
              </div>
            )}

            {/* Messages */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px 40px', display: 'flex', flexDirection: 'column', gap: 4 }}>
              {messages.length === 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 12, color: '#9ca3af' }}>
                  <Avatar name={activeLabel} size={72} rgb={activeRgb} />
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#1c1c2e', marginBottom: 4 }}>{activeLabel}</div>
                    <div style={{ fontSize: '0.83rem', color: '#9ca3af' }}>Say hi and start the conversation ✌️</div>
                  </div>
                </div>
              )}

              {messages.map((msg, idx) => {
                const isOwn    = msg.sender === 'user';
                const prev     = messages[idx - 1];
                // Show sender name for non-own messages, or always in group chats
                const showName = !isOwn && (prev?.sender !== 'ai' || idx === 0 || isGroup);

                const isRegen  = regeneratingIds.has(msg.id);

                return (
                  <div key={msg.id ?? idx} style={{
                    display: 'flex', flexDirection: 'column',
                    alignItems: isOwn ? 'flex-end' : 'flex-start',
                    marginTop: (!prev || prev.sender !== msg.sender) ? 12 : 2,
                    position: 'relative',
                  }}>
                    {showName && (
                      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, marginBottom: 2 }}>
                        <Avatar name={msg.senderName || activeLabel} size={26} rgb={activeRgb} />
                        <span style={{ fontSize: '0.72rem', color: '#9ca3af', marginBottom: 2 }}>{msg.senderName}</span>
                      </div>
                    )}

                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingLeft: !isOwn && !showName ? 34 : 0 }}>
                      <MsgBubble
                        msg={msg}
                        isOwn={isOwn}
                        isRegenerating={isRegen}
                        onClick={(m) => setSelectedMsg(m)}
                        onDelete={onDeleteMessage}
                        partialModels={!msg.analysis && msg.sender === 'user' ? partialModels : null}
                      />

                      {msg.analysis && !isRegen && onRegenerateAnalysis && (
                        <button
                          onClick={() => onRegenerateAnalysis(msg.id)}
                          title="Re-run ML pipeline"
                          style={{ background: 'none', border: '1px solid rgba(0,0,0,.10)', borderRadius: '50%', width: 24, height: 24, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', color: '#9ca3af', flexShrink: 0, opacity: 0, transition: 'opacity .15s' }}
                          className="regen-btn"
                        >↺</button>
                      )}
                    </div>
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </div>

            {/* Input bar */}
            <div style={{ padding: '12px 20px', borderTop: '1px solid #efefef', background: '#ffffff' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 10,
                border: `1.5px solid ${dominantRgb ? `rgba(${dominantRgb},.35)` : 'rgba(0,0,0,.14)'}`,
                borderRadius: 28, padding: '4px 6px 4px 16px',
                transition: 'border-color .4s',
                background: dominantRgb ? `rgba(${dominantRgb},.04)` : 'transparent',
              }}>
                <button style={{ background: 'none', border: 'none', fontSize: '1.3rem', cursor: 'pointer', lineHeight: 1, padding: 0, flexShrink: 0 }}>😊</button>
                <textarea
                  ref={inputRef} rows={1}
                  placeholder="Message…"
                  value={inputValue}
                  onChange={e => setInputValue(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); } }}
                  style={{ flex: 1, background: 'none', border: 'none', outline: 'none', resize: 'none', fontSize: '0.95rem', color: '#1c1c2e', fontFamily: 'inherit', padding: '9px 0', lineHeight: 1.4, maxHeight: 100, overflowY: 'auto' }}
                />
                {inputValue.trim() ? (
                  <button onClick={onSend} style={{ background: 'none', border: 'none', cursor: 'pointer', fontWeight: 700, fontSize: '0.9rem', color: '#0077ff', padding: '6px 10px', flexShrink: 0 }}>Send</button>
                ) : (
                  <button onClick={() => { setInputValue('❤️'); setTimeout(() => onSend(), 50); }}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.4rem', padding: '4px 6px', flexShrink: 0, transition: 'transform .15s' }}
                    onMouseOver={e => e.currentTarget.style.transform = 'scale(1.2)'}
                    onMouseOut={e => e.currentTarget.style.transform = 'scale(1)'}
                  >❤️</button>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── Right Panel: Telemetry / Analysis Drawer ──────────────────── */}
      {rightPanelOpen && (
        <div style={{
          width: 280, borderLeft: '1px solid #efefef',
          display: 'flex', flexDirection: 'column',
          background: '#ffffff', flexShrink: 0,
          animation: 'igModalIn .2s ease both',
        }}>
          {selectedMsg ? (
            <AnalysisDrawer
              msg={selectedMsg}
              onClose={() => setSelectedMsg(null)}
              onFeedbackSent={() => {}}
            />
          ) : (
            <TelemetryPanel
              processing={processing}
              lastAnalysis={currentAnalysis}
              partialModels={partialModels}
            />
          )}
        </div>
      )}

      <style>{`
        @keyframes igMsgIn {
          from { opacity:0; transform:translateY(6px) scale(.96); }
          to   { opacity:1; transform:translateY(0) scale(1); }
        }
        @keyframes igModalIn {
          from { opacity:0; transform:scale(.94) translateY(8px); }
          to   { opacity:1; transform:scale(1) translateY(0); }
        }
        @keyframes regen-spin {
          to { transform: rotate(360deg); }
        }
        div:hover > div > .regen-btn { opacity: 1 !important; }
      `}</style>
    </div>
  );
}
