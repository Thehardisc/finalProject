import React, { useState, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { EmotionPalette } from '../components/EmotionPalette';
import { getEmotionCSSVars } from '../utils/emotionColorUtils';
import { EMOTION_COLOR_CONFIG as CFG } from '../utils/emotionColorConfig';
import AnalysisDrawer        from '../components/AnalysisDrawer';
import DemoRunner            from '../components/DemoRunner';
import PixelFace, { toEkman } from '../components/PixelFace';
import EmotionArcChart       from '../components/EmotionArcChart';
import CDMStateGraph         from '../components/CDMStateGraph';
import AmbientOrb            from '../components/AmbientOrb';
import TelemetryPanel        from '../components/TelemetryPanel';
import PlutchikWheel         from '../visualizations/PlutchikWheel';
import EmotionIntelligencePanel from '../components/EmotionIntelligencePanel';
import '../styles/ig-theme.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const CDM_STATE_LABELS = [
  'Neutral','Warmth','Praise','Help Request','Humor','Tension',
  'Conflict','Argument','Withdrawal','Reconciliation','Curiosity',
  'Assertiveness','Empathy','Frustration','Agreement',
];
const cdmLabel = (v) =>
  typeof v === 'number' ? (CDM_STATE_LABELS[v] ?? `State ${v}`) :
  typeof v === 'string' ? v.replace(/_/g, ' ') : '';

const EMOTION_EMOJI = {
  joy:'😄', happiness:'😊', sadness:'😢', anger:'😠', fear:'😨',
  surprise:'😲', disgust:'🤢', neutral:'😐', excitement:'🤩',
  love:'❤️', caring:'🤗', gratitude:'🙏', amusement:'😂',
  admiration:'🤩', annoyance:'😒', confusion:'😕', curiosity:'🤔',
  desire:'😍', disappointment:'😞', disapproval:'👎', embarrassment:'😳',
  grief:'😭', nervousness:'😬', optimism:'🌟', pride:'😌',
  realization:'💡', relief:'😮‍💨', remorse:'😔', approval:'👍',
};

const SUGGESTIONS = {
  anger:          ["I hear you — can we slow down?", "That wasn't my intention. I'm sorry.", "Let's take a breath."],
  sadness:        ["I'm here with you.", "That sounds really hard.", "You're not alone in this."],
  fear:           ["You're safe. Tell me more.", "Let's work through it together.", "I'm listening."],
  grief:          ["I'm so sorry.", "Take all the time you need.", "I'm right here."],
  annoyance:      ["Fair point — I'm listening.", "I'm sorry, let's try again.", "What's really bothering you?"],
  disgust:        ["I understand your frustration.", "Let's find a better path.", "Your feelings are valid."],
  disappointment: ["I'm sorry I let you down.", "What can I do differently?", "Tell me what you need."],
  nervousness:    ["Take a breath — it's okay.", "What's on your mind?", "One step at a time."],
  disapproval:    ["Tell me what's not working.", "I want to understand.", "Fair — let's talk it through."],
  remorse:        ["Please don't be too hard on yourself.", "We all make mistakes.", "I'm here."],
};

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

function buildDemoMessages(myName, otherName) {
  return [
    {
      id: 'demo-1', sender: 'ai', senderName: otherName,
      text: "Sure, totally fine with that 🙄",
      analysis: {
        type: 'analysis',
        data: {
          id: 'demo-1', raw_text: "Sure, totally fine with that 🙄",
          final_dominant_emotion: 'annoyance', final_valence: -0.42,
          meta_confidence: 0.81, sarcasm_score: 0.91,
          bert_emotions: [
            { label: 'annoyance',   score: 0.48 },
            { label: 'disgust',     score: 0.22 },
            { label: 'disapproval', score: 0.15 },
            { label: 'neutral',     score: 0.10 },
            { label: 'joy',         score: 0.05 },
          ],
          logic_map: { VADER: 0.17, BERT: 0.33, GoEmotions: 0.43, Context: 0.07 },
          gate_weights_alpha: [0.19, 0.34, 0.47],
          ekman_group: 'disgust',
          context_snapshot: {
            cur_valence: -0.42, prev_emotion: 'neutral',
            topic_resonance: 0.31, volatility: 0.72, ce_available: true,
            cdm_current_state: 'CONFLICT_RISE', cdm_entry_abruptness: 0.68, cdm_residency: 0.14,
          },
          lstm_trajectory: {
            model_available: true, top_predicted: 'disgust',
            predicted_next: { disgust: 0.38, annoyance: 0.29, disapproval: 0.18, anger: 0.09, neutral: 0.06 },
          },
          dynamics:  { inertia: 0.62,  contagion: -0.35 },
          appraisal: { novelty: 0.72, goal_congruence: -0.58, coping: 0.28 },
        },
      },
    },
    {
      id: 'demo-2', sender: 'user', senderName: myName,
      text: "Wait, are you being sarcastic right now?",
      analysis: {
        type: 'analysis',
        data: {
          id: 'demo-2', raw_text: "Wait, are you being sarcastic right now?",
          final_dominant_emotion: 'confusion', final_valence: -0.18,
          meta_confidence: 0.74, sarcasm_score: 0.12,
          bert_emotions: [
            { label: 'confusion',   score: 0.44 },
            { label: 'nervousness', score: 0.26 },
            { label: 'surprise',    score: 0.18 },
            { label: 'neutral',     score: 0.08 },
            { label: 'anger',       score: 0.04 },
          ],
          logic_map: { VADER: 0.18, BERT: 0.32, GoEmotions: 0.42, Context: 0.08 },
          gate_weights_alpha: [0.21, 0.37, 0.42],
          ekman_group: 'surprise',
          context_snapshot: {
            cur_valence: -0.30, prev_emotion: 'annoyance',
            topic_resonance: 0.28, volatility: 0.65, ce_available: true,
            cdm_current_state: 'TENSION', cdm_entry_abruptness: 0.55, cdm_residency: 0.20,
          },
          lstm_trajectory: {
            model_available: true, top_predicted: 'anger',
            predicted_next: { anger: 0.31, annoyance: 0.27, disgust: 0.22, fear: 0.12, neutral: 0.08 },
          },
          dynamics:  { inertia: 0.45,  contagion: 0.58 },
          appraisal: { novelty: 0.65, goal_congruence: -0.30, coping: 0.42 },
        },
      },
    },
    {
      id: 'demo-3', sender: 'ai', senderName: otherName,
      text: "Yes I was. I'm sorry — I've been stressed and that wasn't fair to you.",
      analysis: {
        type: 'analysis',
        data: {
          id: 'demo-3', raw_text: "Yes I was. I'm sorry — I've been stressed and that wasn't fair to you.",
          final_dominant_emotion: 'remorse', final_valence: -0.28,
          meta_confidence: 0.88, sarcasm_score: 0.04,
          bert_emotions: [
            { label: 'remorse',      score: 0.52 },
            { label: 'sadness',      score: 0.24 },
            { label: 'caring',       score: 0.12 },
            { label: 'neutral',      score: 0.07 },
            { label: 'nervousness',  score: 0.05 },
          ],
          logic_map: { VADER: 0.22, BERT: 0.35, GoEmotions: 0.38, Context: 0.05 },
          gate_weights_alpha: [0.25, 0.39, 0.36],
          ekman_group: 'sadness',
          context_snapshot: {
            cur_valence: -0.20, prev_emotion: 'confusion',
            topic_resonance: 0.44, volatility: 0.58, ce_available: true,
            cdm_current_state: 'REPAIR', cdm_entry_abruptness: 0.40, cdm_residency: 0.25,
          },
          lstm_trajectory: {
            model_available: true, top_predicted: 'caring',
            predicted_next: { caring: 0.35, relief: 0.25, gratitude: 0.20, joy: 0.12, neutral: 0.08 },
          },
          dynamics:  { inertia: -0.28, contagion: -0.18 },
          appraisal: { novelty: 0.55, goal_congruence: 0.42, coping: 0.61 },
        },
      },
    },
    {
      id: 'demo-4', sender: 'user', senderName: myName,
      text: "Thank you for saying that. I appreciate your honesty.",
      analysis: {
        type: 'analysis',
        data: {
          id: 'demo-4', raw_text: "Thank you for saying that. I appreciate your honesty.",
          final_dominant_emotion: 'gratitude', final_valence: 0.61,
          meta_confidence: 0.91, sarcasm_score: 0.03,
          bert_emotions: [
            { label: 'gratitude',   score: 0.56 },
            { label: 'relief',      score: 0.22 },
            { label: 'caring',      score: 0.12 },
            { label: 'joy',         score: 0.07 },
            { label: 'approval',    score: 0.03 },
          ],
          logic_map: { VADER: 0.30, BERT: 0.25, GoEmotions: 0.38, Context: 0.07 },
          gate_weights_alpha: [0.31, 0.30, 0.39],
          ekman_group: 'joy',
          context_snapshot: {
            cur_valence: 0.20, prev_emotion: 'remorse',
            topic_resonance: 0.60, volatility: 0.42, ce_available: true,
            cdm_current_state: 'RECOVERY', cdm_entry_abruptness: 0.30, cdm_residency: 0.30,
          },
          lstm_trajectory: {
            model_available: true, top_predicted: 'joy',
            predicted_next: { joy: 0.38, optimism: 0.28, relief: 0.18, caring: 0.10, gratitude: 0.06 },
          },
          dynamics:  { inertia: 0.22,  contagion: 0.71 },
          appraisal: { novelty: 0.32, goal_congruence: 0.78, coping: 0.82 },
        },
      },
    },
    {
      id: 'demo-5', sender: 'ai', senderName: otherName,
      text: "Me too. Let's start fresh — want to grab coffee tomorrow? ☕",
      analysis: {
        type: 'analysis',
        data: {
          id: 'demo-5', raw_text: "Me too. Let's start fresh — want to grab coffee tomorrow? ☕",
          final_dominant_emotion: 'optimism', final_valence: 0.72,
          meta_confidence: 0.87, sarcasm_score: 0.02,
          bert_emotions: [
            { label: 'optimism',  score: 0.38 },
            { label: 'relief',    score: 0.28 },
            { label: 'approval',  score: 0.18 },
            { label: 'gratitude', score: 0.10 },
            { label: 'joy',       score: 0.06 },
          ],
          logic_map: { VADER: 0.25, BERT: 0.28, GoEmotions: 0.37, Context: 0.10 },
          gate_weights_alpha: [0.27, 0.32, 0.41],
          ekman_group: 'joy',
          context_snapshot: {
            cur_valence: 0.72, prev_emotion: 'caring',
            topic_resonance: 0.71, volatility: 0.34, ce_available: true,
            cdm_current_state: 'RESOLUTION', cdm_entry_abruptness: 0.22, cdm_residency: 0.41,
          },
          lstm_trajectory: {
            model_available: true, top_predicted: 'joy',
            predicted_next: { joy: 0.36, optimism: 0.28, excitement: 0.18, relief: 0.11, gratitude: 0.07 },
          },
          dynamics:  { inertia: 0.48,  contagion: 0.84 },
          appraisal: { novelty: 0.28, goal_congruence: 0.86, coping: 0.91 },
        },
      },
    },
  ];
}

// ── Avatar ─────────────────────────────────────────────────────────────────────

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
          background: '#22c55e', border: '2px solid #05060F',
        }} />
      )}
    </div>
  );
}

// ── MsgBubble ──────────────────────────────────────────────────────────────────

function formatMsgTime(ts) {
  if (ts == null) return null;
  const d = new Date(ts);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function MsgBubble({ msg, isOwn, onClick, isRegenerating, onDelete, showTimestamp = false, showConfidence = true }) {
  const [hovered, setHovered] = useState(false);
  const dom          = msg.analysis?.data?.final_dominant_emotion;
  const domRgb       = dom ? emotionRgb(dom) : null;
  const sarcasmScore = msg.analysis?.data?.sarcasm_score ?? 0;
  const isAnalyzing  = !msg.analysis && msg.sender === 'user' && !isRegenerating;
  const confidence   = msg.analysis?.data?.meta_confidence;
  const timeLabel    = showTimestamp ? formatMsgTime(msg.timestamp) : null;

  // Flat bubbles: a slightly-elevated neutral surface (var(--ig-bub-in) — visible
  // against both the dark and light chat backgrounds) with a thin emotion-coloured
  // outline on BOTH members' bubbles. No gradient fill.
  const lineColor = domRgb ? `rgb(${domRgb})` : 'var(--ig-bord2)';

  return (
    <motion.div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => msg.analysis && onClick?.(msg)}
      initial={{ opacity: 0, y: 10, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      style={{
        display: 'flex', flexDirection: 'column',
        alignItems: isOwn ? 'flex-end' : 'flex-start',
        minWidth: 0, maxWidth: '100%',
        marginBottom: 3, cursor: msg.analysis ? 'pointer' : 'default',
      }}
    >
      <motion.div
        whileTap={{ scale: 0.97 }}
        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        style={{
          maxWidth: '100%', width: 'fit-content',
          padding: '10px 14px',
          borderRadius: isOwn ? '22px 22px 6px 22px' : '22px 22px 22px 6px',
          background: 'var(--ig-bub-in)',
          border: `1.5px solid ${isAnalyzing ? 'var(--ig-bord2)' : lineColor}`,
          fontSize: '0.94rem', lineHeight: 1.5,
          color: 'var(--ig-txt)', overflowWrap: 'break-word',
          boxShadow: '0 1px 2px rgba(0,0,0,.10)',
          position: 'relative', overflow: 'hidden',
          outline: hovered && msg.analysis ? `2px solid rgba(${domRgb || '109,40,217'},.40)` : 'none',
          outlineOffset: 1, transition: 'outline-color .15s',
          minWidth: isAnalyzing ? 120 : 0,
        }}
      >
        {isAnalyzing ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <div style={{ height: 11, borderRadius: 6, width: '85%', background: 'linear-gradient(90deg,rgba(0,119,255,.08) 25%,rgba(0,119,255,.18) 50%,rgba(0,119,255,.08) 75%)', backgroundSize: '200% 100%', animation: 'shimmer 1.8s ease-in-out infinite' }} />
            <div style={{ height: 11, borderRadius: 6, width: '60%', background: 'linear-gradient(90deg,rgba(0,119,255,.08) 25%,rgba(0,119,255,.18) 50%,rgba(0,119,255,.08) 75%)', backgroundSize: '200% 100%', animation: 'shimmer 1.8s ease-in-out infinite .3s' }} />
            <div style={{ marginTop: 2, fontSize: '0.72rem', color: 'var(--ig-txt3)', fontStyle: 'italic' }}>Analyzing…</div>
          </div>
        ) : isRegenerating ? (
          <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--ig-txt3)', fontStyle: 'italic', fontSize: '0.87rem' }}>
            <span style={{ width: 12, height: 12, border: '2px solid rgba(0,0,0,.10)', borderTopColor: '#6b7280', borderRadius: '50%', display: 'inline-block', animation: 'regen-spin .7s linear infinite' }} />
            Re-analyzing…
          </span>
        ) : (
          <>
            {msg.text}
            {sarcasmScore > 0.75 && <span className="bubble__sarcasm-icon" title={`Sarcasm ${Math.round(sarcasmScore * 100)}%`}>⟳</span>}
          </>
        )}
      </motion.div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 3 }}>
        {dom && dom.toLowerCase() !== 'neutral' && !isAnalyzing && (
          <div style={{ fontSize: '0.66rem', color: domRgb ? `rgb(${domRgb})` : '#9ca3af', fontWeight: 600, letterSpacing: '.04em', opacity: .75, textTransform: 'capitalize' }}>
            {dom}{showConfidence && confidence != null && (
              <span style={{ opacity: .8, fontWeight: 500 }}> · {Math.round(confidence * 100)}%</span>
            )}
          </div>
        )}
        {timeLabel && !isAnalyzing && (
          <div style={{ fontSize: '0.62rem', color: 'var(--ig-txt3)' }}>{timeLabel}</div>
        )}
        {msg.analysis && hovered && (
          <div style={{ fontSize: '0.62rem', color: 'var(--ig-txt3)' }}>click to inspect ↗</div>
        )}
        {isOwn && hovered && onDelete && (
          <button onClick={e => { e.stopPropagation(); onDelete(msg.id); }}
            style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '0.65rem', color: '#ef4444', padding: '0 2px', lineHeight: 1, opacity: 0.7 }}
            title="Delete"
          >✕</button>
        )}
      </div>
    </motion.div>
  );
}

// ── SuggestedResponses ─────────────────────────────────────────────────────────

function SuggestedResponses({ dominant, onSelect }) {
  const suggestions = SUGGESTIONS[dominant?.toLowerCase()] || [];
  if (!suggestions.length) return null;
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 380, damping: 28 }}
      style={{ padding: '8px 20px', display: 'flex', gap: 8, flexWrap: 'wrap', flexShrink: 0 }}
    >
      {suggestions.map((s, i) => (
        <motion.button
          key={i}
          initial={{ opacity: 0, scale: 0.86 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: 'spring', stiffness: 400, damping: 28, delay: i * 0.07 }}
          whileTap={{ scale: 0.92 }}
          onClick={() => onSelect(s)}
          style={{
            background: 'rgba(109,40,217,0.12)', border: '1px solid rgba(109,40,217,0.30)',
            borderRadius: 99, padding: '6px 14px', fontSize: '0.78rem', fontWeight: 500,
            color: 'rgba(167,139,250,0.88)', cursor: 'pointer',
          }}
        >{s}</motion.button>
      ))}
    </motion.div>
  );
}

// ── GroupModal ─────────────────────────────────────────────────────────────────

function GroupModal({ globalUsers, currentUser, onConfirm, onClose }) {
  const [groupName,  setGroupName]  = useState('');
  const [selected,   setSelected]   = useState([]);
  const [search,     setSearch]     = useState('');
  const [error,      setError]      = useState('');
  const [submitting, setSubmitting] = useState(false);

  const eligible = globalUsers.filter(u =>
    u.user_id !== currentUser?.user_id &&
    u.display_name?.toLowerCase().includes(search.toLowerCase())
  );
  const toggle = (uid) => setSelected(prev => prev.includes(uid) ? prev.filter(x => x !== uid) : [...prev, uid]);

  const handleSubmit = async () => {
    if (!groupName.trim()) { setError('Group name is required.'); return; }
    if (selected.length === 0) { setError('Add at least one member.'); return; }
    setSubmitting(true);
    try { await onConfirm(groupName.trim(), selected); onClose(); }
    catch (e) { setError(e?.response?.data?.detail || e.message || 'Failed.'); }
    finally { setSubmitting(false); }
  };

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(0,0,0,.20)', display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{ background: 'var(--ig-surf)', border: '1px solid var(--ig-bord)', borderRadius: 20, width: 420, maxHeight: '80vh', display: 'flex', flexDirection: 'column', boxShadow: '0 12px 40px rgba(0,0,0,.12)', overflow: 'hidden', animation: 'igModalIn .22s cubic-bezier(.34,1.2,.64,1) both' }}>
        <div style={{ padding: '20px 20px 0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <button onClick={onClose} style={{ background: 'none', border: 'none', fontSize: '1.4rem', cursor: 'pointer', color: 'var(--ig-txt3)' }}>✕</button>
          <span style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--ig-txt)' }}>New Group</span>
          <div style={{ width: 24 }} />
        </div>
        <div style={{ padding: '14px 20px 0' }}>
          <input autoFocus placeholder="Group name…" value={groupName} onChange={e => setGroupName(e.target.value)}
            style={{ width: '100%', boxSizing: 'border-box', background: 'var(--ig-surf2)', border: '1px solid var(--ig-bord2)', borderRadius: 12, padding: '10px 14px', fontSize: '0.93rem', color: 'var(--ig-txt)', outline: 'none' }} />
        </div>
        <div style={{ padding: '10px 20px 4px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'var(--ig-surf2)', borderRadius: 12, padding: '10px 14px' }}>
            <svg width="14" height="14" fill="none" stroke="#9ca3af" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <input placeholder="Search people…" value={search} onChange={e => setSearch(e.target.value)}
              style={{ background: 'none', border: 'none', outline: 'none', fontSize: '0.90rem', color: 'var(--ig-txt)', flex: 1 }} />
          </div>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '6px 0 8px' }}>
          {eligible.map(u => (
            <button key={u.user_id} onClick={() => toggle(u.user_id)}
              style={{ width: '100%', background: selected.includes(u.user_id) ? 'rgba(109,40,217,0.07)' : 'none', border: 'none', display: 'flex', alignItems: 'center', gap: 12, padding: '10px 20px', cursor: 'pointer', textAlign: 'left' }}
              onMouseOver={e => { if (!selected.includes(u.user_id)) e.currentTarget.style.background = 'var(--ig-surf3)'; }}
              onMouseOut={e => { if (!selected.includes(u.user_id)) e.currentTarget.style.background = 'none'; }}
            >
              <Avatar name={u.display_name} size={36} rgb="91,138,106" />
              <span style={{ flex: 1, fontWeight: 500, fontSize: '0.90rem', color: 'var(--ig-txt)' }}>{u.display_name}</span>
              <div style={{ width: 20, height: 20, borderRadius: '50%', border: `2px solid ${selected.includes(u.user_id) ? '#7c3aed' : '#d1d5db'}`, background: selected.includes(u.user_id) ? '#7c3aed' : 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.65rem', color: '#fff' }}>
                {selected.includes(u.user_id) ? '✓' : ''}
              </div>
            </button>
          ))}
          {eligible.length === 0 && <div style={{ padding: '20px', textAlign: 'center', color: 'var(--ig-txt3)', fontSize: '0.85rem' }}>No users found.</div>}
        </div>
        {error && <div style={{ padding: '0 20px 8px', fontSize: '0.80rem', color: '#dc2626' }}>{error}</div>}
        <div style={{ padding: '12px 20px 20px', borderTop: '1px solid var(--ig-bord)', display: 'flex', gap: 10 }}>
          <button onClick={onClose} style={{ flex: 1, background: 'var(--ig-surf2)', color: 'var(--ig-txt2)', border: 'none', borderRadius: 12, padding: '11px', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem' }}>Cancel</button>
          <button onClick={handleSubmit} disabled={submitting}
            style={{ flex: 1, background: 'linear-gradient(135deg,#4c1d95,#6d28d9)', color: '#fff', border: 'none', borderRadius: 12, padding: '11px', cursor: submitting ? 'wait' : 'pointer', fontWeight: 700, fontSize: '0.88rem', opacity: submitting ? 0.7 : 1 }}>
            {submitting ? 'Creating…' : `Create (${selected.length})`}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── CrisisAlert ────────────────────────────────────────────────────────────────

function CrisisAlert({ onDismiss }) {
  return (
    <div style={{
      margin: '8px 20px 0', borderRadius: 12,
      background: 'rgba(192,57,43,0.07)', border: '1px solid rgba(192,57,43,0.20)',
      padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 10,
      flexShrink: 0,
    }}>
      <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#C0392B', flexShrink: 0, animation: 'pulseGlow 2s ease infinite' }} />
      <span style={{ flex: 1, fontSize: '0.80rem', color: 'rgba(252,165,165,0.90)', lineHeight: 1.4, fontWeight: 500 }}>
        High emotional intensity detected — this conversation may need extra care.
      </span>
      <button onClick={onDismiss} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(139,34,32,0.50)', fontSize: '0.85rem' }}>✕</button>
    </div>
  );
}

// ── EmotionalArcStrip ──────────────────────────────────────────────────────────

function EmotionalArcStrip({ messages, onSelectMsg }) {
  const stripRef = useRef(null);

  const analyzed = useMemo(() =>
    messages.filter(m => m.analysis?.data?.final_valence != null),
    [messages]
  );

  useEffect(() => {
    if (stripRef.current) stripRef.current.scrollLeft = stripRef.current.scrollWidth;
  }, [analyzed.length]);

  if (analyzed.length < 2) return null;

  const DOT_SPACING = 18;
  const H = 36;
  const W = Math.max(analyzed.length * DOT_SPACING + 20, 200);
  const toX = i => 10 + i * DOT_SPACING;
  const toY = v => H / 2 - v * (H / 2 - 5);

  const pts = analyzed.map((m, i) => [toX(i), toY(m.analysis.data.final_valence)]);

  let pathD = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 1; i < pts.length; i++) {
    const cpx = (pts[i - 1][0] + pts[i][0]) / 2;
    pathD += ` C${cpx},${pts[i - 1][1]} ${cpx},${pts[i][1]} ${pts[i][0]},${pts[i][1]}`;
  }

  return (
    <div
      ref={stripRef}
      style={{
        height: 52, flexShrink: 0,
        overflowX: 'auto', overflowY: 'hidden',
        borderTop: '1px solid rgba(var(--ig-ink-rgb),0.08)',
        background: 'rgba(var(--ig-ink-rgb),0.06)',
        display: 'flex', alignItems: 'center',
        padding: '0 8px',
        scrollbarWidth: 'none',
      }}
    >
      <svg width={W} height={H} style={{ minWidth: W, overflow: 'visible' }}>
        <line x1={0} y1={H / 2} x2={W} y2={H / 2}
          stroke="rgba(var(--ig-ink-rgb),0.10)" strokeWidth="1" strokeDasharray="3 4" />
        <path d={pathD} fill="none" stroke="rgba(109,40,217,0.40)" strokeWidth="1.5" strokeLinecap="round" />
        {analyzed.map((m, i) => {
          const rgb = EmotionPalette[m.analysis.data.final_dominant_emotion?.toLowerCase()] ?? EmotionPalette.neutral;
          const [cx, cy] = pts[i];
          const isLast = i === analyzed.length - 1;
          return (
            <circle
              key={m.id ?? i}
              cx={cx} cy={cy}
              r={isLast ? 4.5 : 3}
              fill={`rgb(${rgb})`}
              opacity={isLast ? 1 : 0.65}
              style={{ cursor: 'pointer', transition: 'r .15s' }}
              onClick={() => onSelectMsg?.(m)}
            >
              <title>{m.analysis.data.final_dominant_emotion} · {m.text?.slice(0, 40)}</title>
            </circle>
          );
        })}
      </svg>
    </div>
  );
}

// ── LiveAnalysisPanel ──────────────────────────────────────────────────────────

const SECT = {
  fontSize: '0.58rem', fontWeight: 700, color: 'rgba(var(--ig-ink-rgb),0.38)',
  textTransform: 'uppercase', letterSpacing: '.10em',
  marginBottom: 8,
};

function Bar({ value, max = 1, color = '99,102,241', label, right }) {
  const pct = Math.min(1, Math.max(0, value / max)) * 100;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
      {label && <span style={{ fontSize: '0.72rem', color: 'rgba(var(--ig-ink-rgb),0.30)', width: 68, flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>}
      <div style={{ flex: 1, height: 4, borderRadius: 3, background: 'rgba(var(--ig-ink-rgb),0.09)', position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${pct}%`, background: `rgba(${color},0.80)`, borderRadius: 3, transition: 'width .4s ease' }} />
      </div>
      {right && <span style={{ fontSize: '0.68rem', color: 'rgba(var(--ig-ink-rgb),0.38)', width: 30, textAlign: 'right', flexShrink: 0 }}>{right}</span>}
    </div>
  );
}

function BipolarBar({ value, label }) {
  const clamped = Math.max(-1, Math.min(1, value));
  const pct = ((clamped + 1) / 2) * 100;
  const color = clamped >= 0 ? '52,211,153' : '239,68,68';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
      <span style={{ fontSize: '0.72rem', color: 'rgba(var(--ig-ink-rgb),0.30)', width: 68, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 4, borderRadius: 3, background: 'rgba(var(--ig-ink-rgb),0.09)', position: 'relative' }}>
        <div style={{ position: 'absolute', left: '50%', top: 0, width: 1, height: '100%', background: 'rgba(var(--ig-ink-rgb),0.18)' }} />
        <div style={{
          position: 'absolute',
          left: clamped >= 0 ? '50%' : `${pct}%`,
          top: 0, height: '100%',
          width: `${Math.abs(clamped) * 50}%`,
          background: `rgba(${color},0.75)`,
          borderRadius: 3,
          transition: 'left .4s ease, width .4s ease',
        }} />
      </div>
      <span style={{ fontSize: '0.68rem', color: 'rgba(var(--ig-ink-rgb),0.38)', width: 36, textAlign: 'right', flexShrink: 0 }}>
        {clamped >= 0 ? '+' : ''}{clamped.toFixed(2)}
      </span>
    </div>
  );
}

// Waiting state — PlutchikWheel + component list (no emoji icons)
function AnalysisWaiting() {
  const PREVIEW = [
    { dot: '167,139,250', label: 'Dominant Emotion',  sub: '28-class GoEmotions + 7 Ekman' },
    { dot: '96,165,250',  label: 'Trajectory LSTM',   sub: '12-message emotional lookahead' },
    { dot: '74,222,128',  label: 'CDM · 15 States',   sub: 'Conversation Dynamics Machine'  },
    { dot: '251,191,36',  label: 'VAD + Appraisal',   sub: 'valence · arousal · dominance'  },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '16px 14px', boxSizing: 'border-box' }}>
      {/* PlutchikWheel instead of emoji orb */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, marginBottom: 18 }}>
        <div style={{ width: 88, height: 88, opacity: 0.40, '--accent-primary': 'rgba(139,92,246,0.75)', '--glass-border': 'rgba(var(--ig-ink-rgb),0.14)' }}>
          <PlutchikWheel dominantEmotion="neutral" />
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'rgba(var(--ig-ink-rgb),0.70)', marginBottom: 3 }}>Waiting for signal</div>
          <div style={{ fontSize: '0.66rem', color: 'rgba(var(--ig-ink-rgb),0.28)', lineHeight: 1.5 }}>Send a message to activate<br />the emotion pipeline</div>
        </div>
      </div>
      <div style={{ fontSize: '0.55rem', fontWeight: 700, color: 'rgba(var(--ig-ink-rgb),0.22)', letterSpacing: '.10em', textTransform: 'uppercase', marginBottom: 8 }}>What you'll see</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {PREVIEW.map((p, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'rgba(var(--ig-ink-rgb),0.03)', border: '1px solid rgba(var(--ig-ink-rgb),0.06)', borderRadius: 9, padding: '8px 11px' }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: `rgba(${p.dot},0.70)`, flexShrink: 0 }} />
            <div>
              <div style={{ fontSize: '0.71rem', fontWeight: 700, color: `rgba(${p.dot},0.78)` }}>{p.label}</div>
              <div style={{ fontSize: '0.61rem', color: 'rgba(var(--ig-ink-rgb),0.26)', marginTop: 1 }}>{p.sub}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Processing state — TelemetryPanel from git (dark mode)
function AnalysisProcessing({ partialModels = new Set(), lastAnalysis, dark = false }) {
  return (
    <div style={{ padding: '12px 10px' }}>
      <TelemetryPanel processing={true} lastAnalysis={lastAnalysis} partialModels={partialModels} dark={dark} />
    </div>
  );
}

function LiveAnalysisPanel({ currentAnalysis, processing, partialModels = new Set(), messages = [], dark = false, settings = {} }) {
  const showTrajectory  = settings.showTrajectory ?? true;
  const pipelineVerbose = settings.pipelineVerbose ?? false;
  const showOrb         = settings.ambientOrb ?? true;
  const data      = currentAnalysis?.data;
  const dom       = data?.final_dominant_emotion?.toLowerCase() || 'neutral';
  const rgb       = EmotionPalette[dom] || EmotionPalette.neutral;
  const confidence = data?.meta_confidence ?? 0;
  const valenceHistory = useMemo(() =>
    messages.filter(m => m.analysis?.data?.final_valence != null)
      .map(m => m.analysis.data.final_valence).slice(-24),
    [messages]
  );

  const topBert = useMemo(() =>
    [...(data?.bert_emotions || [])].sort((a, b) => b.score - a.score).slice(0, 5),
    [data?.bert_emotions]
  );

  const traj  = data?.lstm_trajectory;
  const snap  = data?.context_snapshot;
  const vad   = data?.vad || {};
  const dyn   = data?.dynamics || {};
  const appr  = data?.appraisal || {};

  const valence    = vad.valence   ?? data?.final_valence ?? 0;
  const arousal    = vad.arousal   ?? snap?.volatility ?? 0;
  const dominance  = vad.dominance ?? 0;
  const inertia    = dyn.inertia   ?? 0;
  const contagion  = dyn.contagion ?? 0;

  const cdmState     = snap?.cdm_current_state;
  const cdmResidency = snap?.cdm_residency;
  const cdmEntropy   = snap?.emotion_entropy;
  const cdmVelocity  = snap?.velocity;

  const topPredicted  = traj?.top_predicted;
  const predictedNext = traj?.predicted_next || {};
  const topTrajConf   = topPredicted ? (predictedNext[topPredicted] ?? 0) : 0;
  const trajPhase     = traj?.phase;

  const gateWeights   = data?.gate_weights_alpha;
  const sarcasmScore  = data?.sarcasm_score ?? 0;
  const GATE_LABELS   = ['VADER', 'BERT', 'GoE', 'VAD', 'Ctx'];
  const GATE_COLORS   = ['250,204,21', '96,165,250', '167,139,250', '74,222,128', '251,113,133'];

  const DIVIDER = { borderTop: '1px solid rgba(var(--ig-ink-rgb),0.07)', margin: '12px 0' };

  if (!data && !processing) return <AnalysisWaiting />;
  if (processing && !data)  return <AnalysisProcessing partialModels={partialModels} lastAnalysis={currentAnalysis} dark={dark} />;

  const confPct = Math.round(confidence * 100);

  return (
    <div style={{ padding: '14px 14px 20px', overflowY: 'auto', flex: 1, minHeight: 0, boxSizing: 'border-box' }}>

      {/* Ambient orb — emotion-reactive glow behind the rail (fixed, bottom-right) */}
      {showOrb && <AmbientOrb valence={valence} volatility={snap?.volatility ?? 0} />}

      {/* ── Emotion Card ─────────────────────────────────────────────────── */}
      <div style={{
        background: `rgba(${rgb},0.08)`,
        border: `1px solid rgba(${rgb},0.22)`,
        borderRadius: 16, padding: '14px 14px 12px',
        marginBottom: 12, position: 'relative', overflow: 'hidden',
      }}>
        {/* Ambient glow blob */}
        <div style={{ position: 'absolute', right: -20, top: -20, width: 80, height: 80, borderRadius: '50%', background: `radial-gradient(circle, rgba(${rgb},0.25) 0%, transparent 70%)`, filter: 'blur(20px)', pointerEvents: 'none' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
          {/* PixelFace — pixel-art face expressing the detected Ekman emotion */}
          <div style={{ flexShrink: 0, lineHeight: 0 }}>
            <PixelFace emotion={dom} scale={4} showLabel={false} />
          </div>
          <div>
            <div style={{ fontSize: '1.0rem', fontWeight: 800, color: `rgb(${rgb})`, textTransform: 'capitalize', letterSpacing: '.01em' }}>{dom}</div>
            <div style={{ fontSize: '0.62rem', color: 'rgba(var(--ig-ink-rgb),0.35)', marginTop: 1 }}>dominant emotion</div>
          </div>
          <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
            <div style={{ fontSize: '1.1rem', fontWeight: 800, color: `rgba(${rgb},0.90)` }}>{confPct}%</div>
            <div style={{ fontSize: '0.58rem', color: 'rgba(var(--ig-ink-rgb),0.28)' }}>confidence</div>
          </div>
        </div>

        {/* Confidence bar */}
        <div style={{ height: 4, borderRadius: 3, background: 'rgba(var(--ig-ink-rgb),0.08)', overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${confPct}%`, background: `rgba(${rgb},0.75)`, borderRadius: 3, transition: 'width .5s ease', boxShadow: `0 0 8px rgba(${rgb},0.40)` }} />
        </div>

        {/* Badges row */}
        <div style={{ display: 'flex', gap: 5, marginTop: 8, flexWrap: 'wrap' }}>
          {data?.ekman_group && (
            <span style={{ fontSize: '0.60rem', fontWeight: 600, background: 'rgba(var(--ig-ink-rgb),0.07)', color: 'rgba(var(--ig-ink-rgb),0.50)', borderRadius: 5, padding: '2px 7px' }}>
              {data.ekman_group}
            </span>
          )}
          {sarcasmScore > 0.25 && (
            <span style={{ fontSize: '0.60rem', fontWeight: 700, background: 'rgba(251,191,36,0.12)', color: 'rgba(251,191,36,0.88)', border: '1px solid rgba(251,191,36,0.22)', borderRadius: 5, padding: '2px 7px' }}>
              ⚡ sarcasm {Math.round(sarcasmScore * 100)}%
            </span>
          )}
          {data?.inversion_applied && (
            <span style={{ fontSize: '0.60rem', fontWeight: 700, background: 'rgba(248,113,113,0.10)', color: 'rgba(248,113,113,0.80)', border: '1px solid rgba(248,113,113,0.20)', borderRadius: 5, padding: '2px 7px' }}>
              ↕ inverted
            </span>
          )}
        </div>
      </div>

      {/* ── Sentiment Breakdown ───────────────────────────────────────────── */}
      {topBert.length > 0 && (
        <>
          <div style={SECT}>Emotion breakdown</div>
          <div style={{ marginBottom: 10 }}>
            {topBert.map(({ label, score }) => {
              const eRgb = EmotionPalette[label?.toLowerCase()] ?? EmotionPalette.neutral;
              return <Bar key={label} value={score} max={1} color={eRgb} label={label} right={score.toFixed(2)} />;
            })}
          </div>
          {/* EmotionArcChart — valence trajectory across conversation */}
          {valenceHistory.length >= 2 && (
            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: '0.55rem', fontWeight: 700, color: 'rgba(var(--ig-ink-rgb),0.28)', letterSpacing: '.10em', textTransform: 'uppercase', marginBottom: 5 }}>
                Valence arc · {valenceHistory.length} msgs
              </div>
              <EmotionArcChart values={valenceHistory} color={rgb} height={54} />
            </div>
          )}
          <div style={DIVIDER} />
        </>
      )}

      {/* ── Trajectory ───────────────────────────────────────────────────── */}
      {showTrajectory && traj?.model_available && topPredicted && (
        <>
          <div style={SECT}>Trajectory prediction</div>
          <div style={{
            background: 'rgba(109,40,217,0.08)',
            border: '1px solid rgba(139,92,246,0.20)',
            borderRadius: 12, padding: '10px 12px', marginBottom: 10,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{ fontSize: '0.88rem', color: 'rgba(167,139,250,0.55)' }}>→</span>
              <span style={{ fontSize: '0.90rem', fontWeight: 700, color: 'rgba(var(--ig-ink-rgb),0.88)', textTransform: 'capitalize' }}>{topPredicted}</span>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 5, alignItems: 'center' }}>
                {trajPhase && (
                  <span style={{ fontSize: '0.58rem', fontWeight: 600, background: 'rgba(var(--ig-ink-rgb),0.07)', color: 'rgba(var(--ig-ink-rgb),0.40)', borderRadius: 5, padding: '2px 6px', textTransform: 'capitalize' }}>{trajPhase.replace(/_/g,' ')}</span>
                )}
                <span style={{ fontSize: '0.68rem', fontWeight: 700, background: 'rgba(109,40,217,0.20)', color: 'rgba(167,139,250,0.88)', borderRadius: 6, padding: '2px 8px' }}>
                  {Math.round(topTrajConf * 100)}%
                </span>
              </div>
            </div>
            {Object.entries(predictedNext).sort((a,b) => b[1]-a[1]).slice(0, 3).map(([label, score]) => {
              const eRgb = EmotionPalette[label?.toLowerCase()] ?? EmotionPalette.neutral;
              return <Bar key={label} value={score} max={1} color={eRgb} label={label} right={score.toFixed(2)} />;
            })}
          </div>
          <div style={DIVIDER} />
        </>
      )}

      {/* ── CDM State — CDMStateGraph from git ───────────────────────────── */}
      {snap && (
        <>
          {/* Override light-theme text colors for dark context */}
          <style>{`.cdm-dark span[style*="374151"]{color:rgba(var(--ig-ink-rgb),.55)!important}.cdm-dark div[style*="rgba(0,0,0,.07)"]{background:rgba(var(--ig-ink-rgb),.08)!important}`}</style>
          <div className="cdm-dark" style={{ marginBottom: 10 }}>
            <CDMStateGraph snapshot={snap} />
          </div>
          <div style={DIVIDER} />
        </>
      )}

      {/* ── VAD + Dynamics ───────────────────────────────────────────────── */}
      <div style={SECT}>Affective dimensions</div>
      <div style={{ marginBottom: 12 }}>
        <BipolarBar value={valence}   label="Valence" />
        <Bar value={Math.max(0, Math.min(1, arousal))} max={1} color="249,115,22" label="Arousal" right={arousal.toFixed(2)} />
        <BipolarBar value={dominance} label="Dominance" />
        {(inertia !== 0 || contagion !== 0) && (
          <>
            <div style={{ height: 1, background: 'rgba(var(--ig-ink-rgb),0.06)', margin: '7px 0' }} />
            <BipolarBar value={inertia}   label="Inertia" />
            <BipolarBar value={contagion} label="Contagion" />
          </>
        )}
      </div>

      {/* ── Appraisal ────────────────────────────────────────────────────── */}
      {(appr.novelty != null || appr.goal_congruence != null || appr.coping != null) && (
        <>
          <div style={DIVIDER} />
          <div style={SECT}>Appraisal (Scherer)</div>
          <div style={{ marginBottom: 12 }}>
            {appr.novelty       != null && <Bar value={appr.novelty}        max={1} color="192,132,252" label="Novelty"    right={appr.novelty.toFixed(2)} />}
            {appr.goal_congruence != null && <Bar value={appr.goal_congruence} max={1} color="52,211,153"  label="Goal Cong." right={appr.goal_congruence.toFixed(2)} />}
            {appr.coping        != null && <Bar value={appr.coping}         max={1} color="96,165,250"  label="Coping"     right={appr.coping.toFixed(2)} />}
          </div>
        </>
      )}

      {/* ── Gate weights ─────────────────────────────────────────────────── */}
      {pipelineVerbose && gateWeights && gateWeights.length >= 3 && (
        <>
          <div style={DIVIDER} />
          <div style={SECT}>Model gate weights</div>
          <div style={{ display: 'flex', gap: 3, marginBottom: 6, height: 28, borderRadius: 8, overflow: 'hidden' }}>
            {gateWeights.slice(0, 5).map((w, i) => {
              const total = gateWeights.slice(0, 5).reduce((s, v) => s + (v || 0), 0) || 1;
              const pct   = ((w || 0) / total) * 100;
              return (
                <div key={i} title={`${GATE_LABELS[i]}: ${(w * 100).toFixed(0)}%`}
                  style={{ flex: pct, background: `rgba(${GATE_COLORS[i]},0.35)`, borderRadius: 0, transition: 'flex .5s ease', minWidth: pct > 2 ? 0 : 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {pct > 12 && <span style={{ fontSize: '0.55rem', fontWeight: 700, color: `rgb(${GATE_COLORS[i]})` }}>{GATE_LABELS[i]}</span>}
                </div>
              );
            })}
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {gateWeights.slice(0, 5).map((w, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                <div style={{ width: 6, height: 6, borderRadius: 2, background: `rgba(${GATE_COLORS[i]},0.70)` }} />
                <span style={{ fontSize: '0.58rem', color: 'rgba(var(--ig-ink-rgb),0.35)' }}>{GATE_LABELS[i]} {((w || 0) * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </>
      )}

      {/* ── Logic map fallback ───────────────────────────────────────────── */}
      {pipelineVerbose && !gateWeights && data?.logic_map && Object.keys(data.logic_map).length > 0 && (
        <>
          <div style={DIVIDER} />
          <div style={SECT}>Pipeline weights</div>
          {Object.entries(data.logic_map).map(([k, v]) => (
            <Bar key={k} value={v} max={1} color="99,102,241" label={k} right={(v * 100).toFixed(0) + '%'} />
          ))}
        </>
      )}
    </div>
  );
}

// ── Living Aura ────────────────────────────────────────────────────────────────

// ── Main Dashboard ─────────────────────────────────────────────────────────────

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
  const theme = 'prism';
  const [search,          setSearch]          = useState('');
  const [showCompose,     setShowCompose]      = useState(false);
  const [showGroupModal,  setShowGroupModal]   = useState(false);
  const [composeSearch,   setComposeSearch]    = useState('');
  const [showProfileMenu, setShowProfileMenu]  = useState(false);
  const [selectedMsg,     setSelectedMsg]      = useState(null);
  const [crisisDismissed, setCrisisDismissed]  = useState(false);
  const [memberPanelOpen, setMemberPanelOpen]  = useState(false);
  const [memberError,     setMemberError]      = useState('');
  const [showSettings,    setShowSettings]     = useState(false);
  const [settings, setSettings] = useState(() => {
    try { return JSON.parse(localStorage.getItem('ig_settings') || '{}'); } catch { return {}; }
  });
  const updateSetting = (key, val) => setSettings(prev => {
    const next = { ...prev, [key]: val };
    localStorage.setItem('ig_settings', JSON.stringify(next));
    return next;
  });
  useEffect(() => {   // "Analysis panel" setting applies immediately (button still overrides per-session)
    if (settings.showAnalysisPanel != null) setRightPanelOpen(settings.showAnalysisPanel);
  }, [settings.showAnalysisPanel]);
  const igTheme = settings.theme === 'dark' ? 'dark' : 'light';  // chat light/dark mode (default light)

  const messagesContainerRef = useRef(null);
  const inputRef             = useRef(null);
  const profileMenuRef       = useRef(null);

  useEffect(() => {
    document.documentElement.removeAttribute('data-theme');
    const pageBg = igTheme === 'dark' ? '#0a0c12' : '#ffffff';
    document.body.style.background = pageBg;
    document.documentElement.style.background = pageBg;
  }, [igTheme]);

  useEffect(() => {
    if (!showProfileMenu) return;
    const handler = (e) => {
      if (profileMenuRef.current && !profileMenuRef.current.contains(e.target)) {
        setShowProfileMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [showProfileMenu]);

  useEffect(() => {
    const el = messagesContainerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => {
    if (activeConversationId) inputRef.current?.focus();
  }, [activeConversationId]);

  useEffect(() => {
    setSelectedMsg(null);
    setMemberPanelOpen(false);
    setMemberError('');
    setCrisisDismissed(false);
  }, [activeConversationId]);

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
  const [rightPanelOpen, setRightPanelOpen] = useState(() => settings.showAnalysisPanel ?? true);

  const convLabel  = (conv) => conv.type === 'group' ? conv.name : conv.other_display_name;
  const activeLabel = activeConv ? convLabel(activeConv) : '';

  const isCrisis = (() => {
    const last = messages.filter(m => m.analysis?.data?.final_valence != null).slice(-1)[0];
    if (!last) return false;
    return last.analysis.data.final_valence < -0.55 &&
           (last.analysis.data.context_snapshot?.volatility ?? 0) > 0.70;
  })();

  const suggestionEmotion = (() => {
    const lastOther = messages.filter(m => m.sender !== 'user' && m.analysis?.data?.final_valence != null).slice(-1)[0];
    if (!lastOther || lastOther.analysis.data.final_valence >= -0.35) return null;
    return lastOther.analysis.data.final_dominant_emotion;
  })();

  const handleAddMember = async (userId) => {
    setMemberError('');
    try { await onAddMember(activeConversationId, userId); }
    catch (e) { setMemberError(e?.response?.data?.detail || e.message || 'Failed.'); }
  };

  const handleRemoveMember = async (userId) => {
    setMemberError('');
    try { await onRemoveMember(activeConversationId, userId); }
    catch (e) { setMemberError(e?.response?.data?.detail || e.message || 'Failed.'); }
  };

  return (
    <div data-ig-theme={igTheme} style={{
      display: 'flex', height: '100vh', overflow: 'hidden',
      background: 'var(--ig-bg)', color: 'var(--ig-txt)',
      fontFamily: '-apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif',
    }}>

      {/* ── Compose Modal ──────────────────────────────────────────────── */}
      <AnimatePresence>
      {showCompose && (
        <motion.div
          key="compose-backdrop"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(0,0,0,.20)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onClick={() => { setShowCompose(false); setComposeSearch(''); }}
        >
          <motion.div
            key="compose-card"
            initial={{ opacity: 0, scale: 0.92, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 6 }}
            transition={{ type: 'spring', stiffness: 340, damping: 26 }}
            onClick={e => e.stopPropagation()} style={{ background: 'var(--ig-surf)', border: '1px solid var(--ig-bord)', borderRadius: 20, width: 400, maxHeight: '70vh', display: 'flex', flexDirection: 'column', boxShadow: '0 12px 40px rgba(0,0,0,.12)', overflow: 'hidden' }}>
            <div style={{ padding: '20px 20px 0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <button onClick={() => { setShowCompose(false); setComposeSearch(''); }} style={{ background: 'none', border: 'none', fontSize: '1.4rem', cursor: 'pointer', color: 'var(--ig-txt3)' }}>✕</button>
              <span style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--ig-txt)' }}>New Message</span>
              <div style={{ width: 24 }} />
            </div>
            <div style={{ padding: '14px 20px 0' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'var(--ig-surf2)', borderRadius: 12, padding: '10px 14px' }}>
                <svg width="16" height="16" fill="none" stroke="#9ca3af" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                <input autoFocus placeholder="Search people…" value={composeSearch} onChange={e => setComposeSearch(e.target.value)} style={{ background: 'none', border: 'none', outline: 'none', fontSize: '0.93rem', color: 'var(--ig-txt)', flex: 1 }} />
              </div>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '10px 0 8px' }}>
              {filteredUsers.length === 0 && <div style={{ padding: '20px', textAlign: 'center', color: 'var(--ig-txt3)', fontSize: '0.88rem' }}>No users found.</div>}
              {filteredUsers.map(u => (
                <button key={u.user_id} onClick={() => { onCreateChat(u.user_id); setShowCompose(false); setComposeSearch(''); }}
                  style={{ width: '100%', background: 'none', border: 'none', display: 'flex', alignItems: 'center', gap: 12, padding: '10px 20px', cursor: 'pointer', textAlign: 'left' }}
                  onMouseOver={e => e.currentTarget.style.background = 'var(--ig-surf2)'}
                  onMouseOut={e => e.currentTarget.style.background = 'none'}
                >
                  <Avatar name={u.display_name} size={44} rgb="91,138,106" online={onlineUsers.has(u.user_id)} />
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.93rem', color: 'var(--ig-txt)' }}>{u.display_name}</div>
                    <div style={{ fontSize: '0.78rem', color: onlineUsers.has(u.user_id) ? '#16a34a' : '#9ca3af' }}>
                      {onlineUsers.has(u.user_id) ? 'Active now' : 'Offline'}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
      </AnimatePresence>

      {/* ── Group Modal ─────────────────────────────────────────────────── */}
      {showGroupModal && (
        <GroupModal
          globalUsers={globalUsers}
          currentUser={currentUser}
          onConfirm={onCreateGroup}
          onClose={() => setShowGroupModal(false)}
          dark={false}
        />
      )}

      {/* ── Settings Modal ───────────────────────────────────────────────── */}
      <AnimatePresence>
      {showSettings && (
        <motion.div
          key="settings-backdrop"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          onClick={() => setShowSettings(false)} style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(0,0,0,0.60)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <motion.div
            key="settings-card"
            initial={{ opacity: 0, scale: 0.92, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 6 }}
            transition={{ type: 'spring', stiffness: 340, damping: 26 }}
            onClick={e => e.stopPropagation()} style={{ background: 'var(--ig-surf)', border: '1px solid var(--ig-bord)', borderRadius: 20, width: 420, maxWidth: '90vw', boxShadow: '0 12px 40px rgba(0,0,0,.12)', overflow: 'hidden' }}>
            {/* Header */}
            <div style={{ padding: '20px 24px 16px', borderBottom: '1px solid var(--ig-bord)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ width: 32, height: 32, borderRadius: 9, background: 'rgba(109,40,217,0.08)', border: '1px solid rgba(109,40,217,0.18)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#7c3aed' }}>
                  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                </div>
                <span style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--ig-txt)' }}>Settings</span>
              </div>
              <button onClick={() => setShowSettings(false)} style={{ background: 'var(--ig-surf2)', border: '1px solid var(--ig-bord)', borderRadius: '50%', width: 30, height: 30, color: 'var(--ig-txt2)', cursor: 'pointer', fontSize: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>×</button>
            </div>

            {/* Settings rows */}
            <div style={{ padding: '8px 0 16px' }}>
              {/* Appearance — force light / dark theme */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 24px' }}>
                <div>
                  <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--ig-txt)' }}>Appearance</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--ig-txt2)', marginTop: 2 }}>Force light or dark mode for the chat</div>
                </div>
                <div style={{ display: 'flex', gap: 4, background: 'var(--ig-surf2)', border: '1px solid var(--ig-bord)', borderRadius: 10, padding: 3 }}>
                  {['light', 'dark'].map(opt => {
                    const active = igTheme === opt;
                    return (
                      <button key={opt} onClick={() => updateSetting('theme', opt)}
                        style={{ display: 'flex', alignItems: 'center', gap: 5, border: 'none', cursor: 'pointer',
                          background: active ? 'var(--ig-surf)' : 'transparent',
                          color: active ? 'var(--ig-txt)' : 'var(--ig-txt3)',
                          fontWeight: 600, fontSize: '0.78rem', borderRadius: 8, padding: '5px 12px',
                          boxShadow: active ? '0 1px 4px rgba(0,0,0,.12)' : 'none', transition: 'all .15s' }}>
                        <span style={{ fontSize: '0.85rem' }}>{opt === 'light' ? '☀' : '☾'}</span>{opt === 'light' ? 'Light' : 'Dark'}
                      </button>
                    );
                  })}
                </div>
              </div>
              <div style={{ margin: '4px 24px 8px', borderTop: '1px solid var(--ig-bord)' }} />
              {[
                { key: 'showAnalysisPanel', label: 'Analysis panel', sub: 'Show emotion signal rail when a conversation is open', default: true },
                { key: 'showTimestamps',    label: 'Message timestamps', sub: 'Display send time below each message', default: false },
                { key: 'showConfidence',    label: 'Confidence scores', sub: 'Show meta-learner confidence % on each message', default: true },
                { key: 'ambientOrb',        label: 'Ambient orb', sub: 'Emotion-reactive background color in the analysis panel', default: true },
              ].map(row => {
                const val = settings[row.key] ?? row.default;
                return (
                  <div key={row.key} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 24px', cursor: 'pointer' }}
                    onClick={() => updateSetting(row.key, !val)}
                    onMouseOver={e => e.currentTarget.style.background = 'var(--ig-surf3)'}
                    onMouseOut={e => e.currentTarget.style.background = 'none'}
                  >
                    <div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--ig-txt)' }}>{row.label}</div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--ig-txt2)', marginTop: 2 }}>{row.sub}</div>
                    </div>
                    {/* Toggle pill */}
                    <div style={{ width: 40, height: 22, borderRadius: 11, background: val ? '#6d28d9' : 'var(--ig-bord2)', border: `1px solid ${val ? 'rgba(109,40,217,0.60)' : '#d1d5db'}`, position: 'relative', flexShrink: 0, transition: 'all .18s', marginLeft: 16 }}>
                      <div style={{ position: 'absolute', top: 2, left: val ? 20 : 2, width: 16, height: 16, borderRadius: '50%', background: 'var(--ig-knob)', transition: 'left .18s', boxShadow: '0 1px 4px rgba(0,0,0,.20)' }} />
                    </div>
                  </div>
                );
              })}

              <div style={{ margin: '8px 24px 0', padding: '12px 0 0', borderTop: '1px solid var(--ig-bord)' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--ig-txt3)', letterSpacing: '.08em', marginBottom: 8 }}>PIPELINE</div>
                {[
                  { key: 'pipelineVerbose', label: 'Verbose model labels', sub: 'Show VADER / BERT / GoE labels in the analysis panel', default: false },
                  { key: 'showTrajectory',  label: 'Trajectory prediction', sub: 'Display LSTM next-emotion forecast in the analysis panel', default: true },
                ].map(row => {
                  const val = settings[row.key] ?? row.default;
                  return (
                    <div key={row.key} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0', cursor: 'pointer' }}
                      onClick={() => updateSetting(row.key, !val)}
                      onMouseOver={e => e.currentTarget.style.background = 'var(--ig-surf3)'}
                      onMouseOut={e => e.currentTarget.style.background = 'none'}
                    >
                      <div>
                        <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--ig-txt)' }}>{row.label}</div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--ig-txt2)', marginTop: 2 }}>{row.sub}</div>
                      </div>
                      <div style={{ width: 40, height: 22, borderRadius: 11, background: val ? '#6d28d9' : 'var(--ig-bord2)', border: `1px solid ${val ? 'rgba(109,40,217,0.60)' : '#d1d5db'}`, position: 'relative', flexShrink: 0, transition: 'all .18s', marginLeft: 16 }}>
                        <div style={{ position: 'absolute', top: 2, left: val ? 20 : 2, width: 16, height: 16, borderRadius: '50%', background: 'var(--ig-knob)', transition: 'left .18s', boxShadow: '0 1px 4px rgba(0,0,0,.20)' }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
      </AnimatePresence>

      {/* ── Left Sidebar ─────────────────────────────────────────────────── */}
      <div style={{ width: 320, borderRight: '1px solid var(--ig-bord)', display: 'flex', flexDirection: 'column', background: 'var(--ig-surf)', flexShrink: 0 }}>

        {/* Header */}
        <div style={{ padding: '18px 16px 10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div ref={profileMenuRef} style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', position: 'relative' }}
            onClick={() => setShowProfileMenu(p => !p)}
          >
            <Avatar name={currentUser?.display_name} size={32} rgb="91,138,106" />
            <span style={{ fontWeight: 800, fontSize: '0.95rem', color: 'var(--ig-txt)' }}>{currentUser?.display_name}</span>
            <svg width="12" height="12" fill="none" stroke="#6b7280" strokeWidth="2.5" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>

            <AnimatePresence>
            {showProfileMenu && (
              <motion.div
                key="profile-menu"
                initial={{ opacity: 0, y: -8, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -4, scale: 0.97 }}
                transition={{ type: 'spring', stiffness: 420, damping: 28 }}
                onClick={e => e.stopPropagation()}
                style={{ position: 'absolute', top: '110%', left: 0, zIndex: 100, background: 'var(--ig-surf)', borderRadius: 14, minWidth: 200, boxShadow: '0 8px 32px rgba(0,0,0,.14)', border: '1px solid rgba(0,0,0,.07)', overflow: 'hidden' }}
              >
                <div style={{ padding: '12px 16px', borderBottom: '1px solid rgba(0,0,0,0.07)' }}>
                  <div style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--ig-txt)' }}>{currentUser?.display_name}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--ig-txt2)', marginTop: 2 }}>{currentUser?.email}</div>
                  {currentUser?.role === 'admin' && (
                    <span style={{ display: 'inline-block', marginTop: 4, background: 'linear-gradient(135deg,#f59e0b,#d97706)', color: '#fff', fontSize: '0.60rem', fontWeight: 700, padding: '2px 7px', borderRadius: 5, letterSpacing: '.06em' }}>ADMIN</span>
                  )}
                </div>
                {[
                  { label: 'Analytics',       icon: '✨', action: () => { setShowProfileMenu(false); onGoToAnalytics(); } },
                  onGoToLiveAnalytics && { label: 'Live Analytics', icon: '📊', action: () => { setShowProfileMenu(false); onGoToLiveAnalytics(); } },
                  currentUser?.role === 'admin' && { label: 'Pipeline Inspector', icon: '🔬', action: () => { setShowProfileMenu(false); onGoToAdmin(); }, color: '#7c3aed' },
                ].filter(Boolean).map(item => (
                  <button key={item.label} onClick={item.action}
                    style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: '10px 16px', textAlign: 'left', fontSize: '0.88rem', color: item.color || '#1c1c2e', display: 'flex', alignItems: 'center', gap: 10 }}
                    onMouseOver={e => e.currentTarget.style.background = 'var(--ig-surf2)'}
                    onMouseOut={e => e.currentTarget.style.background = 'none'}
                  >{item.icon} {item.label}</button>
                ))}
                <div style={{ borderTop: '1px solid rgba(0,0,0,0.07)' }}>
                  <button onClick={onLogout}
                    style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: '10px 16px', textAlign: 'left', fontSize: '0.88rem', color: '#dc2626', display: 'flex', alignItems: 'center', gap: 10 }}
                    onMouseOver={e => e.currentTarget.style.background = '#fef2f2'}
                    onMouseOut={e => e.currentTarget.style.background = 'none'}
                  >→ Log out</button>
                </div>
              </motion.div>
            )}
            </AnimatePresence>
          </div>

          <div style={{ display: 'flex', gap: 4 }}>
            <button onClick={() => setShowCompose(true)} title="New Message"
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 7, borderRadius: 8, color: 'var(--ig-txt2)', transition: 'all .12s' }}
              onMouseOver={e => { e.currentTarget.style.background = 'var(--ig-surf2)'; e.currentTarget.style.color = '#1c1c2e'; }}
              onMouseOut={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = '#6b7280'; }}
            >
              <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
            </button>
            <button onClick={() => setShowGroupModal(true)} title="New Group"
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 7, borderRadius: 8, color: 'var(--ig-txt2)', transition: 'all .12s' }}
              onMouseOver={e => { e.currentTarget.style.background = 'var(--ig-surf2)'; e.currentTarget.style.color = '#1c1c2e'; }}
              onMouseOut={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = '#6b7280'; }}
            >
              <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
            </button>
            <button onClick={() => setShowSettings(true)} title="Settings"
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 7, borderRadius: 8, color: 'var(--ig-txt2)', transition: 'all .12s' }}
              onMouseOver={e => { e.currentTarget.style.background = 'var(--ig-surf2)'; e.currentTarget.style.color = '#1c1c2e'; }}
              onMouseOut={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = '#6b7280'; }}
            >
              <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
            </button>
          </div>
        </div>

        {/* Search */}
        <div style={{ padding: '0 12px 10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'var(--ig-surf2)', borderRadius: 10, padding: '8px 12px' }}>
            <svg width="14" height="14" fill="none" stroke="#9ca3af" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <input placeholder="Search…" value={search} onChange={e => setSearch(e.target.value)}
              style={{ background: 'none', border: 'none', outline: 'none', fontSize: '0.88rem', color: 'var(--ig-txt)', flex: 1 }} />
          </div>
        </div>

        {/* Status */}
        <div style={{ padding: '0 16px 8px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontWeight: 700, fontSize: '0.75rem', color: 'var(--ig-txt3)', textTransform: 'uppercase', letterSpacing: '.08em' }}>Conversations</span>
          <span style={{ fontSize: '0.72rem', color: status === 'Live' ? '#16a34a' : '#9ca3af', display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: status === 'Live' ? '#16a34a' : '#9ca3af', display: 'inline-block' }} />
            {status}
          </span>
        </div>

        {/* Conversation list */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {filteredConvs.length === 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '40px 20px', gap: 10, color: 'var(--ig-txt3)' }}>
              <div style={{ fontSize: '1.8rem', opacity: 0.4 }}>💬</div>
              <p style={{ margin: 0, fontSize: '0.85rem', textAlign: 'center' }}>No conversations yet.</p>
              <button onClick={() => setShowCompose(true)} style={{ background: 'linear-gradient(135deg,#4c1d95,#6d28d9)', color: '#fff', border: 'none', borderRadius: 10, padding: '9px 18px', cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem' }}>
                Start chatting
              </button>
            </div>
          )}
          {filteredConvs.map(conv => {
            const rgb      = emotionRgb(conv.dominant_emotion);
            const isActive = conv.conversation_id === activeConversationId;
            const label    = convLabel(conv);
            const lastMsg  = messages.length > 0 && isActive ? messages[messages.length - 1]?.text : (conv.dominant_emotion || 'Start chatting');
            const traj     = conv.trajectory_direction;
            return (
              <motion.button key={conv.conversation_id}
                layout
                onClick={() => { onSelectConversation(conv.conversation_id); setShowProfileMenu(false); }}
                whileTap={{ scale: 0.98 }}
                transition={{ type: 'spring', stiffness: 500, damping: 35, layout: { duration: 0.2 } }}
                style={{
                  width: '100%', background: isActive ? 'rgba(109,40,217,0.07)' : 'none',
                  border: 'none', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '9px 12px', textAlign: 'left',
                  borderLeft: `3px solid ${isActive ? '#7c3aed' : 'transparent'}`,
                }}
                onMouseOver={e => { if (!isActive) e.currentTarget.style.background = 'var(--ig-surf3)'; }}
                onMouseOut={e => { if (!isActive) e.currentTarget.style.background = 'none'; }}
              >
                <Avatar name={label} size={44} rgb={rgb} online={conv.type !== 'group' && onlineUsers.has(conv.other_user_id)} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                      <span style={{ fontWeight: isActive ? 700 : 600, fontSize: '0.90rem', color: 'var(--ig-txt)' }}>{label}</span>
                      {conv.type === 'group' && (
                        <span style={{ fontSize: '0.60rem', background: 'rgba(109,40,217,0.08)', color: '#7c3aed', fontWeight: 700, padding: '1px 5px', borderRadius: 5 }}>
                          {conv.member_count || 0}
                        </span>
                      )}
                    </div>
                    <span style={{ fontSize: '0.68rem', color: 'var(--ig-txt3)', flexShrink: 0 }}>{timeAgo(conv.last_message_time || conv.conversation_id)}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 2 }}>
                    {conv.dominant_emotion && (
                      <span style={{ width: 5, height: 5, borderRadius: '50%', background: `rgb(${rgb})`, flexShrink: 0, display: 'inline-block' }} />
                    )}
                    <div style={{ fontSize: '0.78rem', color: 'var(--ig-txt3)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                      {typeof lastMsg === 'string' ? lastMsg : '…'}
                    </div>
                  </div>
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* Footer */}
        <div style={{ padding: '10px 12px', borderTop: '1px solid var(--ig-bord)' }}>
          <button onClick={() => setShowCompose(true)} style={{
            width: '100%', background: 'linear-gradient(135deg,#4c1d95,#6d28d9)', color: '#fff', border: 'none',
            borderRadius: 10, padding: '10px', cursor: 'pointer', fontWeight: 700, fontSize: '0.85rem',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
            boxShadow: '0 3px 12px rgba(109,40,217,.45)',
          }}>
            <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            New Message
          </button>
        </div>
      </div>

      {/* ── Center: Chat ──────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'hidden', position: 'relative' }}>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
        {!activeConversationId ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', overflowY: 'auto', padding: '32px 24px' }}>
            {/* Hero */}
            <div style={{ textAlign: 'center', marginBottom: 36 }}>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'rgba(109,40,217,0.06)', border: '1px solid rgba(109,40,217,0.15)', borderRadius: 99, padding: '4px 14px', marginBottom: 18 }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#16a34a', animation: 'pulse 2s ease infinite' }} />
                <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#7c3aed', letterSpacing: '.08em', textTransform: 'uppercase' }}>Pipeline Active</span>
              </div>
              <h2 style={{ margin: '0 0 10px', fontSize: '1.6rem', fontWeight: 800, color: 'var(--ig-txt)', letterSpacing: '-0.02em' }}>Emotion intelligence, live.</h2>
              <p style={{ margin: '0 0 24px', fontSize: '0.90rem', color: 'var(--ig-txt2)', maxWidth: 400, lineHeight: 1.6 }}>
                Select a conversation or start a new one. Every message flows through 4 parallel ML models in under a second.
              </p>
              <button onClick={() => setShowCompose(true)} style={{ background: 'linear-gradient(135deg,#4c1d95,#6d28d9)', color: '#fff', border: 'none', borderRadius: 12, padding: '12px 32px', cursor: 'pointer', fontWeight: 700, fontSize: '0.92rem', boxShadow: '0 4px 20px rgba(109,40,217,0.30)', transition: 'all .2s', fontFamily: 'inherit' }}>
                New Conversation →
              </button>
            </div>
            {/* Feature cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, maxWidth: 580, width: '100%' }}>
              {[
                { svg: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>, label: 'Emotion Detection', desc: '28 GoEmotions classes + 7 Ekman categories fused by a trained meta-learner', color: '124,58,237' },
                { svg: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" viewBox="0 0 24 24"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>, label: 'Trajectory LSTM', desc: 'Predicts where your conversation is headed emotionally — 12-message lookahead', color: '37,99,235' },
                { svg: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" viewBox="0 0 24 24"><circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><line x1="12" y1="7" x2="5" y2="17"/><line x1="12" y1="7" x2="19" y2="17"/></svg>, label: 'CDM · 15 States', desc: 'Tension, Empathy, Conflict, Humor — Hidden Markov Machine tracks it all', color: '22,163,74' },
                { svg: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" viewBox="0 0 24 24"><rect x="2" y="7" width="6" height="10" rx="1"/><rect x="9" y="9" width="6" height="6" rx="1"/><rect x="16" y="5" width="6" height="14" rx="1"/></svg>, label: 'VAD + Sarcasm', desc: 'Valence/Arousal/Dominance + real-time sarcasm classifier per message', color: '217,119,6' },
              ].map((f, i) => (
                <div key={i} style={{ background: 'var(--ig-surf3)', border: '1px solid var(--ig-bord)', borderRadius: 16, padding: '18px 16px', transition: 'all .25s', cursor: 'default' }}>
                  <div style={{ width: 34, height: 34, borderRadius: 10, background: `rgba(${f.color},0.08)`, border: `1px solid rgba(${f.color},0.14)`, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 10, color: `rgb(${f.color})` }}>{f.svg}</div>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: `rgb(${f.color})`, marginBottom: 4 }}>{f.label}</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--ig-txt2)', lineHeight: 1.55 }}>{f.desc}</div>
                </div>
              ))}
            </div>
            {/* Pipeline strip */}
            <div style={{ marginTop: 24, display: 'flex', alignItems: 'center', gap: 0, flexWrap: 'wrap', justifyContent: 'center' }}>
              {['VADER', '→', 'BERT', '→', 'GoE', '→', 'Context', '→', 'Meta'].map((n, i) => (
                <span key={i} style={{ fontSize: '0.65rem', fontWeight: n === '→' ? 400 : 700, color: n === '→' ? '#d1d5db' : '#6b7280', padding: n === '→' ? '0 4px' : '3px 8px', background: n === '→' ? 'none' : '#f3f4f6', borderRadius: 6, letterSpacing: '.04em' }}>{n}</span>
              ))}
              <span style={{ marginLeft: 8, fontSize: '0.60rem', color: 'var(--ig-txt3)' }}>· 116 features · &lt; 1s</span>
            </div>
          </div>
        ) : (
          <>
            {/* Chat header */}
            <div style={{ position: 'relative', borderBottom: '1px solid var(--ig-bord)', flexShrink: 0, background: 'var(--ig-surf)' }}>
            <div style={{ padding: '11px 18px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <Avatar name={activeLabel} size={38} rgb={activeRgb} online={!isGroup && onlineUsers.has(activeConv?.other_user_id)} />
                <div>
                  <div style={{ fontWeight: 700, fontSize: '0.92rem', color: 'var(--ig-txt)', display: 'flex', alignItems: 'center', gap: 5 }}>
                    {activeLabel}
                    {isGroup && <span style={{ fontSize: '0.60rem', background: 'rgba(109,40,217,0.08)', color: '#7c3aed', fontWeight: 700, padding: '1px 6px', borderRadius: 5 }}>GROUP · {activeConv?.member_count || 0}</span>}
                  </div>
                  <div style={{ fontSize: '0.70rem', color: !isGroup && onlineUsers.has(activeConv?.other_user_id) ? '#16a34a' : '#9ca3af' }}>
                    {!isGroup && onlineUsers.has(activeConv?.other_user_id) ? 'Active now' : isGroup ? 'Group chat' : 'Offline'}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
                {isGroup && (
                  <button onClick={() => setMemberPanelOpen(p => !p)} title="Manage members"
                    style={{ background: memberPanelOpen ? 'rgba(109,40,217,0.08)' : 'none', border: `1px solid ${memberPanelOpen ? 'rgba(109,40,217,.30)' : 'var(--ig-bord)'}`, borderRadius: 8, padding: '5px 10px', cursor: 'pointer', fontSize: '0.72rem', color: memberPanelOpen ? '#7c3aed' : '#6b7280', fontWeight: 500, transition: 'all .15s' }}>
                    👥 Members
                  </button>
                )}
                {socketRef && onDemoStart && (
                  <DemoRunner currentUser={currentUser} socketRef={socketRef} onDemoStart={onDemoStart} />
                )}
                {onInjectDemo && (
                  <button
                    onClick={() => onInjectDemo(buildDemoMessages(currentUser.display_name, activeLabel || 'Other'))}
                    title="Inject demo conversation"
                    style={{ background: 'none', border: '1px solid var(--ig-bord)', borderRadius: 8, padding: '5px 10px', cursor: 'pointer', fontSize: '0.70rem', color: 'var(--ig-txt2)', fontWeight: 500, transition: 'all .12s' }}
                    onMouseOver={e => { e.currentTarget.style.background = 'var(--ig-surf2)'; e.currentTarget.style.color = '#1c1c2e'; }}
                    onMouseOut={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = '#6b7280'; }}
                  >⚗ Demo</button>
                )}
                <button onClick={onGoToAnalytics} title="Analytics"
                  style={{ background: 'none', border: 'none', borderRadius: 8, width: 34, height: 34, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--ig-txt2)', transition: 'all .12s' }}
                  onMouseOver={e => { e.currentTarget.style.background = 'var(--ig-surf2)'; e.currentTarget.style.color = '#1c1c2e'; }}
                  onMouseOut={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = '#6b7280'; }}
                >
                  <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                </button>
                <button onClick={() => setRightPanelOpen(p => !p)} title="Toggle analysis panel"
                  style={{ background: rightPanelOpen ? 'rgba(109,40,217,0.08)' : 'none', border: 'none', borderRadius: 8, width: 34, height: 34, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: rightPanelOpen ? '#7c3aed' : '#6b7280', transition: 'all .12s' }}
                  onMouseOver={e => { e.currentTarget.style.background = rightPanelOpen ? 'rgba(109,40,217,0.12)' : '#f3f4f6'; }}
                  onMouseOut={e => { e.currentTarget.style.background = rightPanelOpen ? 'rgba(109,40,217,0.08)' : 'none'; }}
                >
                  <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
                </button>
              </div>
            </div>
            {/* Emotion color strip — 2px bar that pulses with dominant emotion */}
            {currentAnalysis?.data && (() => {
              const d = currentAnalysis.data;
              const eRgb = EmotionPalette[d.final_dominant_emotion?.toLowerCase()] || EmotionPalette.neutral;
              const conf = d.meta_confidence ?? 0;
              return (
                <div style={{ height: 2, background: 'rgba(var(--ig-ink-rgb),0.04)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${conf * 100}%`, background: `rgba(${eRgb},0.70)`, transition: 'width .8s ease, background .6s ease', boxShadow: `0 0 8px rgba(${eRgb},0.60)` }} />
                </div>
              );
            })()}
            </div>

            {/* Member panel */}
            {isGroup && memberPanelOpen && (
              <div style={{ borderBottom: '1px solid var(--ig-bord)', padding: '12px 18px', background: 'var(--ig-surf3)', maxHeight: 200, overflowY: 'auto', flexShrink: 0 }}>
                <div style={{ fontWeight: 600, fontSize: '0.72rem', color: 'var(--ig-txt3)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '.06em' }}>Members ({activeConv?.member_count || 0})</div>
                {(activeConv?.members || []).map(m => {
                  const isMe    = m.user_id === currentUser?.user_id;
                  const isOwner = m.user_id === activeConv?.creator_user_id;
                  return (
                    <div key={m.user_id} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
                      <Avatar name={m.display_name} size={26} rgb="91,138,106" online={onlineUsers.has(m.user_id)} />
                      <span style={{ flex: 1, fontSize: '0.85rem', color: 'var(--ig-txt)' }}>{m.display_name}{isMe ? ' (you)' : ''}</span>
                      {isOwner && <span style={{ fontSize: '0.60rem', color: '#d97706', fontWeight: 700 }}>★</span>}
                      {isCreator && !isMe && (
                        <button onClick={() => handleRemoveMember(m.user_id)}
                          style={{ background: 'none', border: '1px solid rgba(220,38,38,0.20)', borderRadius: 5, padding: '2px 7px', cursor: 'pointer', fontSize: '0.68rem', color: '#dc2626' }}
                        >Remove</button>
                      )}
                    </div>
                  );
                })}
                {isCreator && globalUsers.filter(u => !(activeConv?.members || []).some(m => m.user_id === u.user_id)).map(u => (
                  <button key={u.user_id} onClick={() => handleAddMember(u.user_id)}
                    style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', background: 'none', border: 'none', padding: '4px 0', cursor: 'pointer', textAlign: 'left' }}
                  >
                    <Avatar name={u.display_name} size={22} rgb="91,138,106" />
                    <span style={{ fontSize: '0.82rem', color: 'var(--ig-txt2)' }}>{u.display_name}</span>
                    <span style={{ fontSize: '0.70rem', color: '#7c3aed', marginLeft: 'auto', fontWeight: 600 }}>+ Add</span>
                  </button>
                ))}
                {memberError && <div style={{ fontSize: '0.75rem', color: '#dc2626', marginTop: 5 }}>{memberError}</div>}
              </div>
            )}

            {/* Messages */}
            <div
              ref={messagesContainerRef}
              style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '16px 24px', display: 'flex', flexDirection: 'column', gap: 4 }}
            >
              {messages.length === 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 14 }}>
                  <div style={{ position: 'relative' }}>
                    <div style={{ position: 'absolute', inset: -10, borderRadius: '50%', background: `rgba(${activeRgb},0.06)`, animation: 'pulseGlow 2.5s ease infinite' }} />
                    <Avatar name={activeLabel} size={72} rgb={activeRgb} />
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 800, fontSize: '1.0rem', color: 'var(--ig-txt)', marginBottom: 5 }}>{activeLabel}</div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--ig-txt2)', lineHeight: 1.6 }}>
                      Say hello and start the conversation.<br />
                      <span style={{ color: '#7c3aed' }}>Every message gets full emotion analysis.</span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                    {['👋', '😊', '❤️'].map((em, i) => (
                      <button key={i} onClick={() => { if (onSend && inputValue === em) onSend(); else { setInputValue(em); setTimeout(() => onSend && onSend(), 50); } }}
                        style={{ background: 'var(--ig-surf2)', border: '1px solid var(--ig-bord2)', borderRadius: 12, padding: '8px 14px', cursor: 'pointer', fontSize: '1.1rem', transition: 'all .2s' }}
                        onMouseOver={e => { e.currentTarget.style.background = 'rgba(109,40,217,0.06)'; e.currentTarget.style.borderColor = 'rgba(109,40,217,0.20)'; }}
                        onMouseOut={e => { e.currentTarget.style.background = 'var(--ig-surf2)'; e.currentTarget.style.borderColor = 'var(--ig-bord2)'; }}
                      >{em}</button>
                    ))}
                  </div>
                </div>
              )}
              {messages.map((msg, idx) => {
                const isOwn    = msg.sender === 'user';
                const prev     = messages[idx - 1];
                const showName = !isOwn && (prev?.sender !== 'ai' || idx === 0 || isGroup);
                const isRegen  = regeneratingIds.has(msg.id);
                return (
                  <div key={msg.id ?? idx} style={{ display: 'flex', justifyContent: isOwn ? 'flex-end' : 'flex-start', marginTop: (!prev || prev.sender !== msg.sender) ? 10 : 2 }}>
                    <div style={{ maxWidth: '68%', display: 'flex', flexDirection: 'column', alignItems: isOwn ? 'flex-end' : 'flex-start' }}>
                      {showName && (
                        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, marginBottom: 2 }}>
                          <Avatar name={msg.senderName || activeLabel} size={24} rgb={activeRgb} />
                          <span style={{ fontSize: '0.68rem', color: 'var(--ig-txt3)', marginBottom: 2 }}>{msg.senderName}</span>
                        </div>
                      )}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, paddingLeft: !isOwn && !showName ? 30 : 0 }}>
                        <MsgBubble
                          msg={msg} isOwn={isOwn} isRegenerating={isRegen}
                          showTimestamp={settings.showTimestamps ?? false}
                          showConfidence={settings.showConfidence ?? true}
                          onClick={(m) => setSelectedMsg(m)}
                          onDelete={onDeleteMessage}
                          partialModels={!msg.analysis && msg.sender === 'user' ? partialModels : null}
                          theme={theme}
                        />
                        {msg.analysis && !isRegen && onRegenerateAnalysis && (
                          <button onClick={() => onRegenerateAnalysis(msg.id)} title="Re-run pipeline"
                            style={{ background: 'none', border: '1px solid var(--ig-bord2)', borderRadius: '50%', width: 22, height: 22, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.70rem', color: 'var(--ig-txt3)', flexShrink: 0, opacity: 0, transition: 'opacity .15s' }}
                            className="regen-btn"
                          >↺</button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Crisis alert */}
            <AnimatePresence>
              {isCrisis && !crisisDismissed && (
                <motion.div
                  key="crisis"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                  style={{ overflow: 'hidden' }}
                >
                  <CrisisAlert onDismiss={() => setCrisisDismissed(true)} />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Suggested responses */}
            {suggestionEmotion && SUGGESTIONS[suggestionEmotion?.toLowerCase()] && (
              <SuggestedResponses
                dominant={suggestionEmotion}
                onSelect={text => { setInputValue(text); inputRef.current?.focus(); }}
              />
            )}

            {/* Emotional Arc Strip */}
            <EmotionalArcStrip
              messages={messages}
              onSelectMsg={(m) => setSelectedMsg(m)}
            />

            {/* Input */}
            <div style={{ padding: '8px 18px 12px', borderTop: '1px solid var(--ig-bord)', background: 'var(--ig-surf)', flexShrink: 0 }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                border: `1px solid ${dominantRgb ? `rgba(${dominantRgb},.30)` : 'var(--ig-bord2)'}`,
                borderRadius: 26, padding: '4px 6px 4px 14px',
                transition: 'border-color .4s, background .4s',
                background: dominantRgb ? `rgba(${dominantRgb},0.04)` : '#f9fafb',
              }}>
                <textarea
                  ref={inputRef} rows={1}
                  placeholder="Message…"
                  value={inputValue}
                  onChange={e => setInputValue(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); } }}
                  style={{ flex: 1, background: 'none', border: 'none', outline: 'none', resize: 'none', fontSize: '0.93rem', color: 'var(--ig-txt)', fontFamily: 'inherit', padding: '8px 0', lineHeight: 1.4, maxHeight: 100, overflowY: 'auto' }}
                />
                {inputValue.trim() ? (
                  <motion.button
                    onClick={onSend}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.90 }}
                    transition={{ type: 'spring', stiffness: 500, damping: 22 }}
                    style={{ background: 'linear-gradient(135deg,#5b21b6,#6d28d9)', border: 'none', cursor: 'pointer', fontWeight: 700, fontSize: '0.82rem', color: '#fff', padding: '8px 16px', borderRadius: 20, flexShrink: 0, boxShadow: '0 3px 12px rgba(109,40,217,.35)' }}
                  >Send</motion.button>
                ) : (
                  <motion.button
                    onClick={() => { setInputValue('❤️'); setTimeout(() => onSend(), 50); }}
                    whileTap={{ scale: 1.40 }}
                    whileHover={{ scale: 1.15 }}
                    transition={{ type: 'spring', stiffness: 500, damping: 18 }}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.3rem', padding: '4px 6px', flexShrink: 0 }}
                  >❤️</motion.button>
                )}
              </div>
            </div>
          </>
        )}
        </div>{/* end content wrapper */}
      </div>

      {/* ── Right Rail — toggled via header button ────────────────────────── */}
      <AnimatePresence>
      {activeConversationId && rightPanelOpen && (
        <motion.div
          key="right-rail"
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 300, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 340, damping: 32 }}
          style={{
            flexShrink: 0, position: 'relative',
            borderLeft: '1px solid rgba(var(--ig-ink-rgb),0.07)',
            display: 'flex', flexDirection: 'column',
            background: 'var(--ig-rail-bg)',
            overflow: 'hidden',
          }}
        >
          {/* Rail header */}
          <div style={{ padding: '12px 16px 10px', borderBottom: '1px solid rgba(var(--ig-ink-rgb),0.08)', flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'rgba(var(--ig-ink-rgb),0.38)', textTransform: 'uppercase', letterSpacing: '.10em' }}>
              {selectedMsg ? 'Message X-Ray' : 'Emotional Signal'}
            </span>
            {selectedMsg && (
              <button onClick={() => setSelectedMsg(null)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '0.72rem', color: 'rgba(var(--ig-ink-rgb),0.38)', padding: '2px 6px', borderRadius: 5, transition: 'all .12s' }}
                onMouseOver={e => { e.currentTarget.style.background = 'rgba(var(--ig-ink-rgb),0.06)'; e.currentTarget.style.color = 'var(--ig-txt)'; }}
                onMouseOut={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = 'rgba(var(--ig-ink-rgb),0.38)'; }}
              >← Live</button>
            )}
          </div>

          {/* Rail content */}
          <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {selectedMsg ? (
              <AnalysisDrawer
                msg={selectedMsg}
                onClose={() => setSelectedMsg(null)}
                onFeedbackSent={() => {}}
              />
            ) : (
              <LiveAnalysisPanel
                currentAnalysis={currentAnalysis}
                processing={processing}
                partialModels={partialModels}
                messages={messages}
                dark={igTheme === 'dark'}
                settings={settings}
              />
            )}
          </div>
        </motion.div>
      )}
      </AnimatePresence>

      <style>{`
        @keyframes igMsgIn {
          from { opacity:0; transform:translateY(5px) scale(.97); }
          to   { opacity:1; transform:translateY(0) scale(1); }
        }
        @keyframes igModalIn {
          from { opacity:0; transform:scale(.94) translateY(6px); }
          to   { opacity:1; transform:scale(1) translateY(0); }
        }
        @keyframes shimmer {
          0%   { background-position: -200% 0; }
          100% { background-position:  200% 0; }
        }
        @keyframes pulseGlow {
          0%,100% { opacity:1; }
          50%      { opacity:.4; }
        }
        @keyframes slideUp {
          from { opacity:0; transform:translateY(8px); }
          to   { opacity:1; transform:translateY(0); }
        }
        div:hover > div > .regen-btn { opacity: 1 !important; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: #f3f4f6; }
        ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 2px; }
        ::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
      `}</style>
    </div>
  );
}
