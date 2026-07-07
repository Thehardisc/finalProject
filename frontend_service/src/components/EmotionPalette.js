
const PALETTE_LIGHT = {
  anger:         '207, 45, 34',
  excitement:    '166, 89, 16',
  joy:           '33, 124, 76',
  surprise:      '156, 95, 14',
  fear:          '151, 72, 197',
  annoyance:     '180, 78, 22',
  admiration:    '193, 48, 135',
  amusement:     '127, 108, 14',
  confusion:     '91, 98, 203',
  curiosity:     '108, 114, 23',
  desire:        '206, 42, 70',
  disgust:       '135, 104, 50',
  disapproval:   '180, 77, 54',
  embarrassment: '185, 65, 109',
  approval:      '35, 122, 93',
  caring:        '189, 63, 97',
  gratitude:     '37, 125, 51',
  love:          '203, 39, 110',
  optimism:      '142, 102, 11',
  pride:         '131, 82, 203',
  realization:   '22, 116, 160',
  relief:        '38, 119, 129',
  sadness:       '38, 108, 199',
  disappointment:'60, 113, 154',
  grief:         '76, 104, 189',
  remorse:       '109, 103, 153',
  nervousness:   '168, 60, 190',
  neutral:       '101, 109, 129',
  default:       '96, 103, 122',
};

const PALETTE_DARK = {
  anger:         '240, 104, 95',
  excitement:    '243, 141, 63',
  joy:           '78, 208, 132',
  surprise:      '243, 188, 63',
  fear:          '185, 133, 224',
  annoyance:     '240, 124, 66',
  admiration:    '232, 115, 173',
  amusement:     '238, 205, 80',
  confusion:     '141, 141, 226',
  curiosity:     '198, 214, 86',
  desire:        '233, 106, 129',
  disgust:       '178, 150, 92',
  disapproval:   '210, 126, 118',
  embarrassment: '220, 143, 175',
  approval:      '81, 205, 143',
  caring:        '236, 156, 175',
  gratitude:     '120, 217, 152',
  love:          '238, 99, 159',
  optimism:      '245, 196, 61',
  pride:         '175, 131, 226',
  realization:   '74, 188, 237',
  relief:        '150, 211, 222',
  sadness:       '91, 154, 230',
  disappointment:'114, 158, 197',
  grief:         '125, 145, 202',
  remorse:       '142, 142, 180',
  nervousness:   '201, 134, 223',
  neutral:       '151, 158, 175',
  default:       '147, 158, 189',
};

// re-renders) re-resolves. Falls back to 'light' with no DOM (SSR / tests).
function resolveTheme() {
  if (typeof document !== 'undefined' &&
      document.querySelector('[data-ig-theme="dark"]')) return 'dark';
  try {
    if (typeof localStorage !== 'undefined' &&
        JSON.parse(localStorage.getItem('ig_settings') || '{}').theme === 'dark') return 'dark';
  } catch {}
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

export const EmotionPalette = new Proxy({}, {
  get(_target, prop) {
    return (activeTheme() === 'dark' ? PALETTE_DARK : PALETTE_LIGHT)[prop];
  },
});
