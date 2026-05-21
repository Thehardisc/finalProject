import React, { useMemo, useEffect, useState, useRef } from 'react';
import { blendEmotions, EmotionPalette } from './EmotionPalette';
import '../AquaTheme.css';

// Convert bert_emotions array [{label, score}] → flat dict {label: score}
const toEmotionDict = (bertEmotions) => {
  if (!bertEmotions || !Array.isArray(bertEmotions)) return null;
  return bertEmotions.reduce((acc, e) => { acc[e.label] = e.score; return acc; }, {});
};

// Determine energy level from dominant emotion name
const HIGH_ENERGY = new Set(['anger', 'excitement', 'joy', 'fear', 'surprise', 'annoyance']);
const LOW_ENERGY  = new Set(['sadness', 'grief', 'remorse', 'calm', 'neutral', 'boredom', 'relief']);

const getEnergyLevel = (emotions, dominantEmotion) => {
  if (!emotions) return 0.3;
  const dom = dominantEmotion?.toLowerCase() || '';
  if (HIGH_ENERGY.has(dom)) return (emotions[dom] || 0) * 1.2;
  if (LOW_ENERGY.has(dom))  return (emotions[dom] || 0) * 0.4;
  const highSum = [...HIGH_ENERGY].reduce((s, k) => s + (emotions[k] || 0), 0);
  return Math.min(highSum, 1.0);
};

export const AquaBubble = ({ message, previousMessageEmotion }) => {
  const [phase, setPhase] = useState('drop'); // drop → ripple → settle
  const filterId = useRef(`wf-${message.id ?? Math.random().toString(36).slice(2)}`).current;

  const emotions = useMemo(
    () => toEmotionDict(message.analysis?.data?.bert_emotions),
    [message.analysis]
  );

  const dominantEmotion = message.analysis?.data?.final_dominant_emotion || 'neutral';

  const blendedRgb = useMemo(() => blendEmotions(emotions), [emotions]);
  const bleedRgb   = useMemo(
    () => previousMessageEmotion ? blendEmotions(toEmotionDict(previousMessageEmotion)) : '220, 220, 220',
    [previousMessageEmotion]
  );

  const energy = useMemo(
    () => getEnergyLevel(emotions, dominantEmotion),
    [emotions, dominantEmotion]
  );

  // Phase sequencing: drop (0–900ms) → ripple (900–2700ms) → settle
  useEffect(() => {
    const t1 = setTimeout(() => setPhase('ripple'), 900);
    const t2 = setTimeout(() => setPhase('settle'), 2700);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [message.id]);

  // SVG turbulence params per phase and energy
  const turbulence = useMemo(() => {
    if (phase === 'drop')   return { freq: '0.08 0.15', scale: 22, dur: '0.9s' };
    if (phase === 'ripple') {
      return energy > 0.5
        ? { freq: '0.05 0.10', scale: 16, dur: '3s' }
        : { freq: '0.015 0.03', scale: 7, dur: '5s' };
    }
    // settle
    return energy > 0.5
      ? { freq: '0.03 0.06', scale: 10, dur: '8s' }
      : { freq: '0.008 0.015', scale: 4, dur: '12s' };
  }, [phase, energy]);

  // Animate values for feTurbulence
  const animFrom = `${turbulence.freq}`;
  const animMid  = energy > 0.5 ? '0.06 0.12' : '0.01 0.02';
  const animTo   = `${turbulence.freq}`;

  const isUser = message.sender === 'user';

  return (
    <div
      className={`spatial-message-container ${isUser ? 'user' : 'ai'}`}
      style={{
        '--bleed-glow':         `rgba(${bleedRgb}, 0.18)`,
        '--active-emotion-rgb': blendedRgb,
        '--active-emotion-sat': `rgb(${blendedRgb})`,
        alignSelf:  isUser ? 'flex-end' : 'flex-start',
        textAlign:  isUser ? 'right' : 'left',
        width: '100%',
      }}
    >
      {/* Hidden SVG filter — one per bubble */}
      <svg className="svg-filter-defs" aria-hidden="true">
        <defs>
          <filter id={filterId} x="-20%" y="-20%" width="140%" height="140%">
            <feTurbulence
              type="fractalNoise"
              baseFrequency={turbulence.freq}
              numOctaves="3"
              result="noise"
            >
              <animate
                attributeName="baseFrequency"
                dur={turbulence.dur}
                values={`${animFrom}; ${animMid}; ${animTo}`}
                repeatCount="indefinite"
              />
            </feTurbulence>
            <feDisplacementMap
              in="SourceGraphic"
              in2="noise"
              scale={turbulence.scale}
              xChannelSelector="R"
              yChannelSelector="G"
            />
          </filter>
        </defs>
      </svg>

      <div className={`aqua-bubble aqua-bubble--${isUser ? 'user' : 'ai'}`}
           style={{ filter: `url(#${filterId})` }}>
        <div className="liquid-glass-surface">
          {/* Sender name */}
          {message.senderName && (
            <span className="bubble-sender">{message.senderName}</span>
          )}

          {/* Message text */}
          <p className="sf-pro-text">{message.text}</p>

          {/* Pending analysis shimmer */}
          {!message.analysis && (
            <span className="analysis-pending">analyzing...</span>
          )}
        </div>
      </div>

      {/* Dominant emotion label */}
      {dominantEmotion && dominantEmotion !== 'neutral' && (
        <div className="emotion-label" style={{ color: `rgba(${blendedRgb}, 0.75)` }}>
          {dominantEmotion}
        </div>
      )}
    </div>
  );
};
