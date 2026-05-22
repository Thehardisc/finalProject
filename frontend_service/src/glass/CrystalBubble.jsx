import React, { useState, useEffect, useRef } from 'react';
import { blendEmotions } from '../components/EmotionPalette';

// Crystal Glass message bubble — pure CSS spring drop, ZERO SVG filters.
export function CrystalBubble({ message, onFeedback, allEmotions }) {
  const isUser = message.sender === 'user';
  const mountedRef = useRef(false);

  useEffect(() => { mountedRef.current = true; }, []);

  const emotionData = message.analysis?.data?.bert_emotions;
  let emotionDict = {};
  if (emotionData && Array.isArray(emotionData)) {
    emotionData.forEach(({ label, score }) => { emotionDict[label] = score; });
  }

  const blendedRgb = Object.keys(emotionDict).length > 0
    ? blendEmotions(emotionDict)
    : '88, 86, 214';

  const dominant = message.analysis?.data?.final_dominant_emotion;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: isUser ? 'flex-end' : 'flex-start',
      width: '100%',
      marginBottom: '10px',
    }}>
      <div style={{
        '--bubble-rgb': blendedRgb,
        animation: 'crystalDrop 0.52s cubic-bezier(0.34, 1.2, 0.64, 1) both',
        background: isUser
          ? `rgba(${blendedRgb}, 0.13)`
          : 'rgba(255, 255, 255, 0.84)',
        backdropFilter: 'blur(20px) saturate(180%)',
        WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        border: `1.5px solid rgba(${blendedRgb}, ${isUser ? '0.32' : '0.16'})`,
        borderRadius: isUser ? '20px 20px 6px 20px' : '20px 20px 20px 6px',
        padding: '12px 16px',
        maxWidth: '72%',
        minWidth: '80px',
        boxShadow: `
          0 4px 20px rgba(${blendedRgb}, 0.16),
          0 1px 0 rgba(255,255,255,0.95) inset,
          0 0 0 6px rgba(${blendedRgb}, 0.04)
        `,
        position: 'relative',
        overflow: 'hidden',
        transition: 'box-shadow 0.4s ease',
      }}>
        {/* Specular highlight — rigid glass top edge */}
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0,
          height: '38%',
          background: 'linear-gradient(180deg, rgba(255,255,255,0.58) 0%, transparent 100%)',
          borderRadius: 'inherit',
          pointerEvents: 'none',
        }} />

        {/* Sender */}
        <div style={{
          fontSize: '0.70rem',
          fontWeight: 700,
          color: `rgb(${blendedRgb})`,
          mixBlendMode: 'color-burn',
          opacity: 0.7,
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          marginBottom: '4px',
          position: 'relative',
          zIndex: 1,
        }}>
          {message.senderName}
        </div>

        {/* Message text */}
        <div style={{
          color: `rgb(${blendedRgb})`,
          mixBlendMode: 'color-burn',
          fontSize: '0.96rem',
          lineHeight: 1.55,
          fontWeight: 500,
          position: 'relative',
          zIndex: 1,
        }}>
          {message.text}
        </div>

        {/* Emotion tag */}
        {dominant && dominant.toLowerCase() !== 'neutral' && (
          <div style={{
            marginTop: '6px',
            fontSize: '0.68rem',
            color: `rgb(${blendedRgb})`,
            mixBlendMode: 'color-burn',
            opacity: 0.55,
            letterSpacing: '0.05em',
            textTransform: 'capitalize',
            position: 'relative',
            zIndex: 1,
          }}>
            {dominant}
          </div>
        )}

        {/* Loading shimmer when analysis pending */}
        {!message.analysis && (
          <div style={{
            position: 'absolute',
            bottom: 6, right: 10,
            width: '8px', height: '8px',
            borderRadius: '50%',
            background: `rgba(${blendedRgb}, 0.4)`,
            animation: 'pulseGlow 1.2s ease-in-out infinite',
          }} />
        )}
      </div>
    </div>
  );
}
