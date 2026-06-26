/**
 * EmotionPalette.js
 * All 27 GoEmotions + neutral → precise Apple-style vibrant RGB strings.
 * Used by WaterMessageBubble and AquaBubble for blended color computation.
 *
 * Color design rules:
 *   • Values are RGB strings: "R, G, B"  (used inside rgba(..., alpha))
 *   • Each color maps to the emotional quality, not a literal hue
 *   • Saturated enough to survive mix-blend-mode: color-burn on white glass
 *   • Neutral uses deep indigo — gray disappears under color-burn
 */

export const EmotionPalette = {
  // ── High Energy / Tension ─────────────────────────────────────────────
  anger:        '255, 45,  42',   // Apple Red
  excitement:   '255, 126,  0',   // Vivid Orange
  joy:          '52,  199,  89',  // Spring Green
  surprise:     '255, 196,   0',  // Brilliant Amber
  fear:         '162,  67, 220',  // Electric Violet
  annoyance:    '255, 110,   0',  // Deep Orange

  // ── Medium Energy ─────────────────────────────────────────────────────
  admiration:   '255,  85, 165',  // Hot Pink
  amusement:    '255, 183,   0',  // Gold
  confusion:    '88,   86, 214',  // Indigo
  curiosity:    '255, 210,   0',  // Warm Yellow
  desire:       '220,  16,  54',  // Crimson
  disgust:      '120,  60,  14',  // Dark Amber-Brown
  disapproval:  '200,  75,  75',  // Muted Red
  embarrassment:'210,  90, 140',  // Rose

  // ── Positive / Connective ─────────────────────────────────────────────
  approval:     '46,  204, 113',  // Emerald
  caring:       '255, 175, 195',  // Soft Coral Pink
  gratitude:    '130, 240, 140',  // Mint
  love:         '255,  20, 130',  // Deep Pink
  optimism:     '255, 207,   0',  // Golden Yellow
  pride:        '124,  32, 226',  // Royal Purple

  // ── Reflective / Cognitive ────────────────────────────────────────────
  realization:  '0,   185, 255',  // Sky Blue
  relief:       '150, 215, 235',  // Pale Cyan

  // ── Low Energy / Melancholic ──────────────────────────────────────────
  sadness:      '0,   122, 255',  // Ocean Blue
  disappointment:'60, 122, 175',  // Steel Blue
  grief:        '20,   20, 110',  // Midnight Navy
  remorse:      '90,   90, 120',  // Slate Mauve

  // ── Calm / Baseline ───────────────────────────────────────────────────
  nervousness:  '180,  75, 205',  // Orchid

  /*
   * Neutral uses deep indigo, NOT gray.
   * Pure gray (220,220,220) vanishes under color-burn on white canvas.
   * Indigo gives visible, calm presence without emotional charge.
   */
  neutral:      '88,   86, 214',  // Indigo

  // Fallback
  default:      '100, 120, 200',
};

const SKIP_PATTERN = /^(vader_|sentiment_|emphasis_|dominant_emotion)/;

/**
 * blendEmotions
 * Takes a flat { emotion: weight } dict and returns a weighted-average
 * RGB string "R, G, B" suitable for rgba(...) or rgb(...).
 */
export const blendEmotions = (emotionWeights) => {
  if (!emotionWeights || typeof emotionWeights !== 'object') {
    return EmotionPalette.neutral;
  }

  let r = 0, g = 0, b = 0, total = 0;

  for (const [emotion, weight] of Object.entries(emotionWeights)) {
    if (SKIP_PATTERN.test(emotion)) continue;
    if (typeof weight !== 'number' || weight <= 0) continue;

    const rgb = EmotionPalette[emotion] || EmotionPalette.default;
    const [cr, cg, cb] = rgb.split(',').map(Number);

    r += cr * weight;
    g += cg * weight;
    b += cb * weight;
    total += weight;
  }

  if (total === 0) return EmotionPalette.neutral;

  return `${Math.round(r / total)}, ${Math.round(g / total)}, ${Math.round(b / total)}`;
};

/**
 * blendEmotionsGradient
 * Returns a CSS linear-gradient() string from the top 2–3 emotions by weight.
 * Angle defaults to 135deg. Output is ready for use in `background:` property.
 */
export const blendEmotionsGradient = (emotionWeights, angle = '135deg') => {
  const fallback = `linear-gradient(${angle}, rgb(88, 86, 214), rgb(100, 120, 200))`;

  if (!emotionWeights || typeof emotionWeights !== 'object') return fallback;

  const entries = Object.entries(emotionWeights)
    .filter(([k, v]) => !SKIP_PATTERN.test(k) && typeof v === 'number' && v > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  if (entries.length === 0) return fallback;

  if (entries.length === 1) {
    const rgb = EmotionPalette[entries[0][0]] || EmotionPalette.default;
    return `linear-gradient(${angle}, rgb(${rgb}), rgba(${rgb}, 0.55))`;
  }

  const stops = entries.map(([emotion], i) => {
    const rgb = EmotionPalette[emotion] || EmotionPalette.default;
    const pct = Math.round((i / (entries.length - 1)) * 100);
    return `rgb(${rgb}) ${pct}%`;
  });

  return `linear-gradient(${angle}, ${stops.join(', ')})`;
};
