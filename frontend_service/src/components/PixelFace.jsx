import React from 'react';

// GoEmotions / BERT label → Ekman 7
const TO_EKMAN = {
  anger: 'anger', annoyance: 'anger', disapproval: 'anger',
  disgust: 'disgust', contempt: 'disgust',
  fear: 'fear', nervousness: 'fear',
  joy: 'joy', amusement: 'joy', excitement: 'joy', love: 'joy',
  optimism: 'joy', pride: 'joy', admiration: 'joy', gratitude: 'joy',
  caring: 'joy', relief: 'joy', desire: 'joy', approval: 'joy',
  sadness: 'sadness', grief: 'sadness', disappointment: 'sadness',
  remorse: 'sadness', embarrassment: 'sadness',
  surprise: 'surprise', confusion: 'surprise', curiosity: 'surprise',
  neutral: 'neutral', realization: 'neutral',
};

// 12×12 logical grid — each entry is [col, row]
const OUTLINE = [
  [3,0],[4,0],[5,0],[6,0],[7,0],[8,0],
  [2,1],[9,1],
  [1,2],[10,2],
  [1,3],[10,3],
  [0,4],[11,4],
  [0,5],[11,5],
  [0,6],[11,6],
  [0,7],[11,7],
  [1,8],[10,8],
  [1,9],[10,9],
  [2,10],[9,10],
  [3,11],[4,11],[5,11],[6,11],[7,11],[8,11],
];

const FEATURES = {
  brows: {
    neutral:  [[3,2],[4,2],[7,2],[8,2]],
    joy:      [[3,2],[4,2],[7,2],[8,2]],
    sadness:  [[2,3],[3,2],[8,2],[9,3]],
    anger:    [[3,2],[4,3],[7,3],[8,2]],
    fear:     [[2,2],[3,2],[8,2],[9,2],[4,1],[7,1]],
    disgust:  [[2,1],[3,2],[7,2],[8,2]],
    surprise: [[2,1],[3,1],[4,1],[7,1],[8,1],[9,1]],
  },
  eyes: {
    neutral:  [[3,4],[4,4],[7,4],[8,4]],
    joy:      [[3,4],[4,4],[7,4],[8,4]],
    sadness:  [[3,4],[4,4],[7,4],[8,4]],
    anger:    [[3,4],[4,4],[7,4],[8,4]],
    fear:     [[3,3],[4,3],[3,4],[4,4],[7,3],[8,3],[7,4],[8,4]],
    disgust:  [[3,4],[4,4],[7,4],[8,4]],
    surprise: [[3,3],[4,3],[3,4],[4,4],[7,3],[8,3],[7,4],[8,4]],
  },
  mouth: {
    neutral:  [[4,8],[5,8],[6,8],[7,8]],
    joy:      [[3,7],[8,7],[4,8],[7,8],[5,9],[6,9]],
    sadness:  [[4,7],[5,7],[6,7],[7,7],[3,8],[8,8]],
    anger:    [[4,7],[5,7],[6,7],[7,7],[3,8],[8,8]],
    fear:     [[4,7],[5,7],[6,7],[7,7],[3,8],[8,8],[4,9],[5,9],[6,9],[7,9]],
    disgust:  [[4,8],[5,7],[6,7],[7,8],[8,9]],
    surprise: [[4,7],[5,7],[6,7],[7,7],[3,8],[8,8],[4,9],[5,9],[6,9],[7,9]],
  },
};

const EMO_COLOR = {
  neutral:  '#94a3b8',
  joy:      '#fbbf24',
  sadness:  '#60a5fa',
  anger:    '#f87171',
  fear:     '#c084fc',
  disgust:  '#34d399',
  surprise: '#fb923c',
};

function buildBoxShadow(ekman) {
  const face  = EMO_COLOR[ekman] || '#94a3b8';
  const dark  = '#0f172a';
  return [
    ...OUTLINE.map(([x, y]) => `${x}px ${y}px 0 0 ${face}`),
    ...FEATURES.brows[ekman].map(([x, y]) => `${x}px ${y}px 0 0 ${dark}`),
    ...FEATURES.eyes[ekman].map(([x, y])  => `${x}px ${y}px 0 0 ${dark}`),
    ...FEATURES.mouth[ekman].map(([x, y]) => `${x}px ${y}px 0 0 ${dark}`),
  ].join(', ');
}

export function toEkman(emotion) {
  return TO_EKMAN[emotion?.toLowerCase()] ?? 'neutral';
}

export default function PixelFace({ emotion, scale = 5, showLabel = true }) {
  const ekman     = toEkman(emotion);
  const boxShadow = buildBoxShadow(ekman);
  const sizePx    = 12 * scale;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
      <div style={{ width: sizePx, height: sizePx, position: 'relative', flexShrink: 0 }}>
        <div
          key={ekman}
          style={{
            width: 1,
            height: 1,
            position: 'absolute',
            top: 0,
            left: 0,
            transform: `scale(${scale})`,
            transformOrigin: 'top left',
            boxShadow,
            animation: 'pfIn 0.24s steps(4, end)',
          }}
        />
      </div>
      {showLabel && (
        <span style={{
          fontSize: '0.65rem',
          fontWeight: 700,
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          color: EMO_COLOR[ekman],
        }}>
          {ekman}
        </span>
      )}
      <style>{`@keyframes pfIn { from { opacity: 0; } to { opacity: 1; } }`}</style>
    </div>
  );
}
