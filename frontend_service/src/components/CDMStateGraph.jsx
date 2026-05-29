import React from 'react';
import { EmotionPalette } from './EmotionPalette';

// GoEmotions label order — must match shared/constants.py EMOTION_LABELS.
// Used only as a fallback when the backend doesn't send top-k names.
const EMOTION_LABELS = [
  'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
  'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
  'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
  'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
  'relief', 'remorse', 'sadness', 'surprise', 'neutral',
];

const RANK_LABEL = ['1st', '2nd', '3rd'];

function colorOf(name) {
  return (name && EmotionPalette[name.toLowerCase()]) || EmotionPalette.neutral;
}

/**
 * CDM v3.0 — Parallel Belief State Machine panel.
 * The conversation is in its TOP-3 emotion states simultaneously; this renders
 * those three states (ranked, with probabilities), the transition catalyst,
 * the inertia/momentum, and how long the top-1 state has held.
 */
export default function CDMStateGraph({ snapshot }) {
  const available = snapshot?.cdm_available ?? false;

  // Prefer the explicit top-3 names/probs from the backend; fall back to the
  // 28-dim belief distribution if only that is present.
  let names = [snapshot?.cdm_top1_name, snapshot?.cdm_top2_name, snapshot?.cdm_top3_name];
  let probs = snapshot?.cdm_top3_probs;

  const belief = snapshot?.cdm_state_probs;
  if ((!names[0] || !probs) && Array.isArray(belief) && belief.length) {
    const ranked = belief
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3);
    names = ranked.map(r => EMOTION_LABELS[r.i]);
    probs = ranked.map(r => r.p);
  }

  const trio = (names || [])
    .map((name, k) => ({ name, prob: probs?.[k] ?? 0 }))
    .filter(r => r.name);

  const momentum  = snapshot?.cdm_momentum ?? 0;
  const residency = snapshot?.cdm_residency ?? 0;
  const catalyst  = snapshot?.cdm_catalyst;
  const top1Rgb   = trio[0] ? colorOf(trio[0].name) : EmotionPalette.neutral;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#374151' }}>
          Conversation Emotion State
        </span>
        <span style={{
          fontSize: '0.62rem', padding: '2px 7px', borderRadius: 20,
          background: available ? 'rgba(20,184,166,.12)' : 'rgba(0,0,0,.06)',
          color: available ? 'rgb(20,184,166)' : '#9ca3af',
          fontWeight: 600,
        }}>
          {available ? 'LIVE' : 'NO DATA'}
        </span>
      </div>

      {available && trio.length > 0 ? (
        <>
          {/* Top-3 simultaneously-active states */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
            {trio.map((row, k) => {
              const rgb = colorOf(row.name);
              const pct = Math.max(0, Math.min(row.prob * 100, 100));
              return (
                <div key={k}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                    <span style={{ fontSize: '0.6rem', fontWeight: 700, color: '#9ca3af', width: 18 }}>
                      {RANK_LABEL[k]}
                    </span>
                    <span style={{
                      width: 9, height: 9, borderRadius: '50%',
                      background: `rgb(${rgb})`, flexShrink: 0,
                      boxShadow: k === 0 ? `0 0 6px rgba(${rgb},.6)` : 'none',
                    }} />
                    <span style={{
                      fontSize: '0.74rem', fontWeight: k === 0 ? 700 : 600,
                      color: k === 0 ? `rgb(${rgb})` : '#4b5563',
                      textTransform: 'capitalize', flex: 1,
                    }}>
                      {row.name}
                    </span>
                    <span style={{ fontSize: '0.7rem', fontWeight: 700, color: `rgb(${rgb})` }}>
                      {pct.toFixed(0)}%
                    </span>
                  </div>
                  <div style={{ height: 4, background: 'rgba(0,0,0,.07)', borderRadius: 2, marginLeft: 26 }}>
                    <div style={{
                      height: '100%', width: `${pct}%`,
                      background: `rgb(${rgb})`, borderRadius: 2,
                      transition: 'width .4s ease',
                    }} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Catalyst chip */}
          {catalyst && (
            <div style={{
              fontSize: '0.64rem', color: '#6b7280',
              background: `rgba(${top1Rgb},.08)`,
              border: `1px solid rgba(${top1Rgb},.18)`,
              borderRadius: 8, padding: '5px 9px', alignSelf: 'flex-start',
            }}>
              ⚡ catalyst:{' '}
              <span style={{ fontWeight: 700, color: `rgb(${top1Rgb})`, textTransform: 'capitalize' }}>
                {catalyst}
              </span>
            </div>
          )}

          {/* Momentum (inertia) bar */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
              <span style={{ fontSize: '0.62rem', color: '#9ca3af' }}>momentum</span>
              <span style={{ fontSize: '0.62rem', color: '#9ca3af' }}>{(momentum * 100).toFixed(0)}%</span>
            </div>
            <div style={{ height: 3, background: 'rgba(0,0,0,.07)', borderRadius: 2 }}>
              <div style={{
                height: '100%', width: `${Math.min(momentum * 100, 100)}%`,
                background: `rgb(${top1Rgb})`, borderRadius: 2, transition: 'width .4s ease',
              }} />
            </div>
          </div>

          {/* Residency */}
          <div style={{ fontSize: '0.62rem', color: '#9ca3af' }}>
            residency: {(residency * 100).toFixed(0)}% in <span style={{ textTransform: 'capitalize' }}>{trio[0].name}</span>
          </div>
        </>
      ) : (
        <div style={{ textAlign: 'center', fontSize: '0.70rem', color: '#9ca3af', padding: '8px 0' }}>
          CDM activates after first message
        </div>
      )}
    </div>
  );
}
