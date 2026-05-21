/**
 * WaterMessageBubble.jsx
 * iOS 26 / Spatial Aqua — Physics-Based Emotion Bubble
 *
 * Architecture:
 *   • 3-phase water-physics state machine: drop → ripple → settle
 *   • Per-bubble inline SVG feTurbulence + feDisplacementMap (unique filter ID)
 *   • High-energy emotions → rapid chaotic ripples
 *   • Low-energy emotions → slow sine-wave undulation
 *   • Visual Bleeding: previous message's blended color bleeds up as a box-shadow
 *   • Zero-Dark Policy: text uses mix-blend-mode:color-burn on white glass
 */

import React, { useMemo, useEffect, useState, useRef } from 'react';
import { blendEmotions } from './EmotionPalette';
import './WaterMessageBubble.css';

// ── Emotion Energy Classification ───────────────────────────────────────────
const ENERGY = {
  HIGH: new Set(['anger', 'excitement', 'joy', 'fear', 'surprise', 'annoyance', 'desire', 'disgust']),
  LOW:  new Set(['sadness', 'grief', 'remorse', 'calm', 'neutral', 'boredom', 'relief', 'disappointment', 'caring']),
};

const computeEnergy = (emotionDict, dominantEmotion) => {
  if (!emotionDict) return 0.25;
  const dom = (dominantEmotion || '').toLowerCase();
  if (ENERGY.HIGH.has(dom)) return Math.min((emotionDict[dom] || 0.5) * 1.3, 1.0);
  if (ENERGY.LOW.has(dom))  return Math.min((emotionDict[dom] || 0.3) * 0.35, 0.4);
  // Weighted sum across all high-energy keys
  return Math.min(
    [...ENERGY.HIGH].reduce((s, k) => s + (emotionDict[k] || 0), 0) * 0.6,
    1.0
  );
};

// ── Turbulence param lookup per phase × energy ──────────────────────────────
const TURBULENCE = {
  drop:   { freq: '0.09 0.17', scale: 24, dur: '0.9s',  mid: '0.07 0.12' },
  ripple: {
    high: { freq: '0.055 0.11', scale: 18, dur: '2.5s', mid: '0.08 0.15' },
    low:  { freq: '0.012 0.025', scale: 6, dur: '6s',   mid: '0.008 0.018' },
  },
  settle: {
    high: { freq: '0.03 0.06',  scale: 10, dur: '9s',   mid: '0.05 0.09' },
    low:  { freq: '0.006 0.012', scale: 3, dur: '14s',  mid: '0.004 0.010' },
  },
};

const getTurbulence = (phase, energy) => {
  if (phase === 'drop') return TURBULENCE.drop;
  const tier = energy > 0.45 ? 'high' : 'low';
  return TURBULENCE[phase][tier];
};

// ── bertEmotions [] → flat dict ─────────────────────────────────────────────
const toDict = (arr) => {
  if (!arr || !Array.isArray(arr)) return null;
  return arr.reduce((acc, { label, score }) => { acc[label] = score; return acc; }, {});
};

// ════════════════════════════════════════════════════════════════════════════
export const WaterMessageBubble = ({ message, previousMessageEmotion }) => {
  const [phase, setPhase] = useState('drop');

  // Stable, unique filter ID per bubble mount
  const filterId = useRef(
    `wmb-${String(message.id ?? '').replace(/[^a-z0-9]/gi, '') || Math.random().toString(36).slice(2)}`
  ).current;

  // ── Emotion data ────────────────────────────────────────────────────────
  const emotionDict = useMemo(
    () => toDict(message.analysis?.data?.bert_emotions),
    [message.analysis]
  );

  const dominant = message.analysis?.data?.final_dominant_emotion || 'neutral';

  const blendedRgb = useMemo(() => blendEmotions(emotionDict),       [emotionDict]);
  const bleedRgb   = useMemo(() => {
    const prev = toDict(previousMessageEmotion);
    return prev ? blendEmotions(prev) : '200, 210, 230';
  }, [previousMessageEmotion]);

  const energy = useMemo(() => computeEnergy(emotionDict, dominant), [emotionDict, dominant]);

  // ── Phase sequencing ────────────────────────────────────────────────────
  useEffect(() => {
    setPhase('drop');
    const t1 = setTimeout(() => setPhase('ripple'), 900);
    const t2 = setTimeout(() => setPhase('settle'), 2800);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [message.id]);

  // ── SVG filter params ────────────────────────────────────────────────────
  const t = useMemo(() => getTurbulence(phase, energy), [phase, energy]);

  const isUser = message.sender === 'user';

  // ── CSS custom props for this bubble ──────────────────────────────────────
  const cssVars = {
    '--bubble-rgb':   blendedRgb,
    '--bleed-rgb':    bleedRgb,
    '--bleed-shadow': `0 -32px 50px rgba(${bleedRgb}, 0.20)`,
  };

  return (
    <div
      className={`wmb-root wmb-root--${isUser ? 'user' : 'ai'}`}
      style={cssVars}
    >
      {/* ── SVG water filter (hidden) ──────────────────────────────────── */}
      <svg className="wmb-filter-host" aria-hidden="true">
        <defs>
          <filter id={filterId} x="-25%" y="-25%" width="150%" height="150%"
                  colorInterpolationFilters="sRGB">
            <feTurbulence
              type="fractalNoise"
              baseFrequency={t.freq}
              numOctaves="4"
              seed="3"
              result="noise"
            >
              <animate
                attributeName="baseFrequency"
                dur={t.dur}
                values={`${t.freq}; ${t.mid}; ${t.freq}`}
                repeatCount="indefinite"
              />
            </feTurbulence>
            <feDisplacementMap
              in="SourceGraphic"
              in2="noise"
              scale={t.scale}
              xChannelSelector="R"
              yChannelSelector="G"
              result="displaced"
            />
          </filter>
        </defs>
      </svg>

      {/* ── Bubble shell ──────────────────────────────────────────────── */}
      <div
        className={`wmb-bubble wmb-bubble--${isUser ? 'user' : 'ai'}`}
        style={{ filter: `url(#${filterId})` }}
      >
        <div className="wmb-glass">
          {/* Sender label */}
          {message.senderName && (
            <span className="wmb-sender">{message.senderName}</span>
          )}

          {/* Message text */}
          <p className="wmb-text">{message.text}</p>

          {/* Pending shimmer */}
          {!message.analysis && (
            <span className="wmb-pending">analyzing…</span>
          )}
        </div>
      </div>

      {/* ── Emotion tag ────────────────────────────────────────────────── */}
      {dominant && dominant.toLowerCase() !== 'neutral' && (
        <div className="wmb-emotion-tag">
          {dominant}
        </div>
      )}
    </div>
  );
};

export default WaterMessageBubble;
