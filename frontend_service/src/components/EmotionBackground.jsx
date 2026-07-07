import { useMemo } from 'react';
import { EmotionPalette } from './EmotionPalette';

/*
 * EmotionBackground — animated, emotion-reactive gradient backdrop.
 *
 * Pure CSS (no WebGL): three drifting blurred blobs plus a slowly rotating
 * conic sheen, colored from EmotionPalette by the live dominant emotion.
 * The root is transparent and absolutely positioned — the host page paints
 * its own base (var(--ig-bg)) and must place content above with z-index.
 */

function rgbOf(emotion) {
  return EmotionPalette[emotion?.toLowerCase()] || EmotionPalette.neutral;
}

// Pull a secondary hue so the field has depth, not one flat tint.
const SECONDARY = {
  neutral: 'pride', joy: 'optimism', anger: 'annoyance', sadness: 'grief',
  fear: 'nervousness', love: 'admiration', surprise: 'excitement',
  disgust: 'disapproval', excitement: 'joy', gratitude: 'caring',
  optimism: 'joy', caring: 'love', pride: 'admiration', curiosity: 'realization',
};

const CSS = `
@keyframes eb-drift1 { 0%,100% { transform: translate(0,0) scale(1); } 33% { transform: translate(8vw,-6vh) scale(1.15); } 66% { transform: translate(-5vw,7vh) scale(0.9); } }
@keyframes eb-drift2 { 0%,100% { transform: translate(0,0) scale(1); } 40% { transform: translate(-7vw,6vh) scale(1.1); } 75% { transform: translate(6vw,-4vh) scale(0.95); } }
@keyframes eb-drift3 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(4vw,5vh) scale(1.12); } }
@keyframes eb-sheen  { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.eb-root { position: absolute; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; background: transparent; }
.eb-blob { position: absolute; border-radius: 50%; filter: blur(90px); transition: background 1200ms ease; will-change: transform; }
.eb-sheen-layer { position: absolute; inset: -50%; opacity: 0.35; animation: eb-sheen 60s linear infinite; transition: background 1200ms ease; }
[data-ig-theme="light"] .eb-blob { opacity: 0.40; }
[data-ig-theme="light"] .eb-sheen-layer { opacity: 0.15; }
@media (prefers-reduced-motion: reduce) {
  .eb-blob, .eb-sheen-layer { animation: none; }
}
`;

export default function EmotionBackground({ emotion = 'neutral' }) {
  const { primary, secondary } = useMemo(() => {
    const p = rgbOf(emotion);
    const s = rgbOf(SECONDARY[emotion?.toLowerCase()] || 'pride');
    return { primary: p, secondary: s };
  }, [emotion]);

  return (
    <div className="eb-root" aria-hidden="true">
      <style>{CSS}</style>

      {/* Slowly rotating conic sheen tints the whole field */}
      <div
        className="eb-sheen-layer"
        style={{
          background: `conic-gradient(from 0deg,
            rgba(${primary},0.22), rgba(${secondary},0.16),
            rgba(${primary},0.10), rgba(${secondary},0.20), rgba(${primary},0.22))`,
        }}
      />

      {/* Drifting emotion-colored blobs */}
      <div className="eb-blob" style={{
        width: '55vw', height: '55vw', top: '-15vw', left: '-12vw',
        background: `radial-gradient(circle, rgba(${primary},0.55) 0%, transparent 68%)`,
        animation: 'eb-drift1 24s ease-in-out infinite',
      }} />
      <div className="eb-blob" style={{
        width: '48vw', height: '48vw', bottom: '-14vw', right: '-10vw',
        background: `radial-gradient(circle, rgba(${secondary},0.50) 0%, transparent 70%)`,
        animation: 'eb-drift2 30s ease-in-out infinite',
      }} />
      <div className="eb-blob" style={{
        width: '40vw', height: '40vw', top: '35%', left: '45%',
        background: `radial-gradient(circle, rgba(${primary},0.30) 0%, transparent 72%)`,
        animation: 'eb-drift3 20s ease-in-out infinite',
      }} />
    </div>
  );
}
