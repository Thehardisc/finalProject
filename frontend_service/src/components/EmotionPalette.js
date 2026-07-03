/**
 * EmotionPalette.js
 * All 27 GoEmotions + neutral → RGB strings, resolved per active UI theme.
 * Both maps are WCAG 2.1 AA-tuned (>=4.5:1 for the small colored labels,
 * >=3:1 for the bars/dots) against their real surfaces:
 *
 *   • PALETTE_LIGHT — deep jewel tones, AA on the white light surfaces
 *     (#ffffff / #f3f4f6). Bright hues (yellows, mint, coral) are deepened so
 *     they don't wash out on white.
 *   • PALETTE_DARK  — luminous tones, AA on the near-black dark surfaces
 *     (#0a0c12 / #14171f / #1c2029). Deep hues (grief, disgust, the purples)
 *     are lifted so they don't sink into the background.
 *
 * Each emotion keeps its HUE identity across both maps (anger red, sadness
 * blue, joy green, …), so it reads as "the same colour", brightness-adapted.
 *
 * `EmotionPalette` is a Proxy: `EmotionPalette[emotion]` returns the value from
 * whichever map matches the active theme (detected from the mounted
 * `[data-ig-theme]` root). Every existing call site — `EmotionPalette[x]`,
 * `blendEmotions`, `blendEmotionsGradient`, `getEmotionRgb` — is theme-aware
 * with no changes. Format: "R, G, B" for `rgb(...)` / `rgba(..., alpha)`.
 * Theme is resolved globally: the app mounts one themed root at a time.
 */

const PALETTE_LIGHT = {
  // ── High energy / tension ─────────────────────────────────
  anger:         '207, 45, 34',
  excitement:    '166, 89, 16',
  joy:           '33, 124, 76',
  surprise:      '156, 95, 14',
  fear:          '151, 72, 197',
  annoyance:     '180, 78, 22',
  // ── Medium energy ─────────────────────────────────────────
  admiration:    '193, 48, 135',
  amusement:     '127, 108, 14',
  confusion:     '91, 98, 203',
  curiosity:     '108, 114, 23',
  desire:        '206, 42, 70',
  disgust:       '135, 104, 50',
  disapproval:   '180, 77, 54',
  embarrassment: '185, 65, 109',
  // ── Positive / connective ─────────────────────────────────
  approval:      '35, 122, 93',
  caring:        '189, 63, 97',
  gratitude:     '37, 125, 51',
  love:          '203, 39, 110',
  optimism:      '142, 102, 11',
  pride:         '131, 82, 203',
  // ── Reflective / cognitive ────────────────────────────────
  realization:   '22, 116, 160',
  relief:        '38, 119, 129',
  // ── Low energy / melancholic ──────────────────────────────
  sadness:       '38, 108, 199',
  disappointment:'60, 113, 154',
  grief:         '76, 104, 189',
  remorse:       '109, 103, 153',
  // ── Calm / baseline ───────────────────────────────────────
  nervousness:   '168, 60, 190',
  neutral:       '101, 109, 129',
  default:       '96, 103, 122',
};

const PALETTE_DARK = {
  // ── High energy / tension ─────────────────────────────────
  anger:         '240, 104, 95',
  excitement:    '243, 141, 63',
  joy:           '78, 208, 132',
  surprise:      '243, 188, 63',
  fear:          '185, 133, 224',
  annoyance:     '240, 124, 66',
  // ── Medium energy ─────────────────────────────────────────
  admiration:    '232, 115, 173',
  amusement:     '238, 205, 80',
  confusion:     '141, 141, 226',
  curiosity:     '198, 214, 86',
  desire:        '233, 106, 129',
  disgust:       '178, 150, 92',
  disapproval:   '210, 126, 118',
  embarrassment: '220, 143, 175',
  // ── Positive / connective ─────────────────────────────────
  approval:      '81, 205, 143',
  caring:        '236, 156, 175',
  gratitude:     '120, 217, 152',
  love:          '238, 99, 159',
  optimism:      '245, 196, 61',
  pride:         '175, 131, 226',
  // ── Reflective / cognitive ────────────────────────────────
  realization:   '74, 188, 237',
  relief:        '150, 211, 222',
  // ── Low energy / melancholic ──────────────────────────────
  sadness:       '91, 154, 230',
  disappointment:'114, 158, 197',
  grief:         '125, 145, 202',
  remorse:       '142, 142, 180',
  // ── Calm / baseline ───────────────────────────────────────
  nervousness:   '201, 134, 223',
  neutral:       '151, 158, 175',
  default:       '147, 158, 189',
};

// ── Active-theme resolution ─────────────────────────────────────────────────
// A single React render reads the palette many times; resolve once per
// synchronous burst and clear on the next microtask so a theme toggle (which
// re-renders) re-resolves. Falls back to 'light' with no DOM (SSR / tests).
function resolveTheme() {
  // Authoritative: the theme of the currently-mounted chat/analytics root.
  if (typeof document !== 'undefined' &&
      document.querySelector('[data-ig-theme="dark"]')) return 'dark';
  // Fallback for first paint (preloaded demo messages can render emotion colours
  // before the themed root commits): the persisted user setting, same key both
  // IGDashboard and AnalyticsPage read.
  try {
    if (typeof localStorage !== 'undefined' &&
        JSON.parse(localStorage.getItem('ig_settings') || '{}').theme === 'dark') return 'dark';
  } catch { /* ignore malformed storage */ }
  return 'light';
}

let _theme = null;
function activeTheme() {
  if (_theme !== null) return _theme;
  _theme = resolveTheme();
  const reset = () => { _theme = null; };
  if (typeof queueMicrotask === 'function') queueMicrotask(reset);
  else Promise.resolve().then(reset);
  return _theme;
}

/**
 * Theme-aware palette. `EmotionPalette[emotion]` → "R, G, B" for the active
 * theme; unknown keys return undefined so `|| EmotionPalette.default` works.
 */
export const EmotionPalette = new Proxy({}, {
  get(_target, prop) {
    return (activeTheme() === 'dark' ? PALETTE_DARK : PALETTE_LIGHT)[prop];
  },
});

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
  const fallback = `linear-gradient(${angle}, rgb(${EmotionPalette.neutral}), rgb(${EmotionPalette.default}))`;

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
