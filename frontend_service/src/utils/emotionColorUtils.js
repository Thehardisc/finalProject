import { EmotionPalette } from '../components/EmotionPalette';
import { EMOTION_COLOR_CONFIG as CFG } from './emotionColorConfig';

export function getEmotionRgb(emotion) {
  if (!emotion) return EmotionPalette.neutral;
  return EmotionPalette[emotion.toLowerCase()] ?? EmotionPalette.neutral;
}

/**
 * Returns CSS Custom Property values to spread onto a bubble element's style.
 * React sets the values; CSS classes do all visual rendering.
 */
export function getEmotionCSSVars(dominantEmotion, confidence) {
  const rgb = getEmotionRgb(dominantEmotion);
  const tint = CFG.TINT_OPACITY_MIN + confidence * (CFG.TINT_OPACITY_MAX - CFG.TINT_OPACITY_MIN);

  return {
    '--emotion-rgb':          rgb,
    '--emotion-color':        `rgb(${rgb})`,
    '--emotion-tint-opacity': tint.toFixed(3),
  };
}
