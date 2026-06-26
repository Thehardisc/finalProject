// Conversational Syntax Highlighting engine — pure JS, no deps.

export const ROLES = {
  'emotion-pos': { label: 'Positive affect', note: 'warmth, joy, gratitude' },
  'emotion-neg': { label: 'Negative affect', note: 'fear, anger, sadness' },
  'question':    { label: 'Question',        note: 'opens, probes, asks' },
  'request':     { label: 'Request',         note: 'asks for action' },
  'agreement':   { label: 'Agreement',       note: 'affirms, aligns' },
  'refusal':     { label: 'Refusal',         note: 'declines, resists' },
  'hedge':       { label: 'Hedge',           note: 'softens, qualifies' },
  'intensifier': { label: 'Intensifier',     note: 'amplifies what follows' },
  'absolute':    { label: 'Absolute',        note: 'always / never / everyone' },
  'negation':    { label: 'Negation',        note: 'flips meaning' },
  'default':     { label: 'Connective',      note: 'structural text' },
};

// emotion words map to a specific GoEmotion (reuse this to wire into your real BERT output)
const EMO_POS = {
  love:'love', loved:'love', loving:'love', adore:'love',
  happy:'joy', glad:'joy', joy:'joy', joyful:'joy', delighted:'joy',
  great:'joy', wonderful:'joy', awesome:'joy', amazing:'admiration',
  good:'joy', fun:'joy', nice:'joy', lovely:'joy', beautiful:'admiration',
  thank:'gratitude', thanks:'gratitude', grateful:'gratitude', appreciate:'gratitude',
  excited:'excitement', thrilled:'excitement', proud:'pride',
  hope:'optimism', hopeful:'optimism', better:'optimism',
  care:'caring', caring:'caring', safe:'relief', relieved:'relief', calm:'relief',
};
const EMO_NEG = {
  hate:'anger', angry:'anger', mad:'anger', furious:'anger', annoyed:'annoyance',
  annoying:'annoyance', frustrated:'annoyance', frustrating:'annoyance',
  scared:'fear', afraid:'fear', terrified:'fear', worried:'nervousness',
  worry:'nervousness', anxious:'nervousness', nervous:'nervousness', stressed:'nervousness',
  overwhelmed:'nervousness', panic:'fear',
  sad:'sadness', unhappy:'sadness', down:'sadness', lonely:'sadness', alone:'sadness',
  hurt:'sadness', hurts:'sadness', crying:'sadness', cry:'sadness', tired:'sadness',
  upset:'disappointment', disappointed:'disappointment',
  sorry:'remorse', regret:'remorse', guilty:'remorse', ashamed:'embarrassment',
  embarrassed:'embarrassment', awful:'disgust', terrible:'disgust', horrible:'disgust',
  disgusting:'disgust', sick:'disgust', bad:'disappointment',
};
const QUESTION  = new Set(['why','what','how','when','where','who','which','whom','whose']);
const REQUEST   = new Set(['please','can','could','would','will','let','need','want','help','should']);
const AGREEMENT = new Set(['yes','yeah','yep','yup','sure','ok','okay','agreed','agree','exactly','right','definitely','certainly','course','fine','deal','understood','got']);
const REFUSAL   = new Set(['no','nope','nah','refuse','wont','stop','enough']);
const HEDGE     = new Set(['maybe','perhaps','possibly','probably','might','guess','think','suppose','seems','seem','kind','sort','kinda','sorta','just','somewhat','slightly','apparently','hopefully','feel','feels','rather','bit']);
const INTENS    = new Set(['very','really','so','super','extremely','incredibly','totally','absolutely','completely','utterly','deeply','truly','too','such','quite','way','most','damn']);
const ABSOLUTE  = new Set(['always','never','everyone','everybody','nobody','everything','nothing','none','all','every','forever','constantly','anyone','anything','ever','whole']);
const NEGATION  = new Set(['not','dont','doesnt','didnt','cant','cannot','wont','isnt','arent','wasnt','werent','havent','hasnt','wouldnt','couldnt','shouldnt','aint','neither','nor','without','no']);

const PHRASES = [
  [['kind','of'],'hedge'],[['sort','of'],'hedge'],[['i','guess'],'hedge'],[['i','think'],'hedge'],
  [['i','suppose'],'hedge'],[['a','bit'],'hedge'],[['a','little'],'hedge'],[['or','something'],'hedge'],
  [['can','you'],'request'],[['could','you'],'request'],[['would','you'],'request'],[['will','you'],'request'],
  [['can','we'],'request'],[['could','we'],'request'],[['let','us'],'request'],[['need','you'],'request'],[['have','to'],'request'],
  [['of','course'],'agreement'],[['for','sure'],'agreement'],[['no','problem'],'agreement'],
  [['you','always'],'absolute'],[['you','never'],'absolute'],[['every','time'],'absolute'],
];

function wordRole(w) {
  if (EMO_NEG[w]) return 'emotion-neg';
  if (EMO_POS[w]) return 'emotion-pos';
  if (NEGATION.has(w)) return 'negation';
  if (ABSOLUTE.has(w)) return 'absolute';
  if (INTENS.has(w))   return 'intensifier';
  if (REQUEST.has(w))  return 'request';
  if (HEDGE.has(w))    return 'hedge';
  if (AGREEMENT.has(w))return 'agreement';
  if (REFUSAL.has(w))  return 'refusal';
  if (QUESTION.has(w)) return 'question';
  return 'default';
}
function hashConf(w, role) {
  let h = 0; const s = w + '|' + role;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  const base = role === 'default' ? 0.45 : 0.70;
  return Math.min(0.98, base + (h % 100) / 100 * 0.28);
}
function emotionFor(w, role) {
  if (role === 'emotion-pos') return EMO_POS[w] || 'joy';
  if (role === 'emotion-neg') return EMO_NEG[w] || 'sadness';
  return null;
}

// classify(text) → [{ text, ws, word|null, role, conf, emotion }]
export function classify(text) {
  if (!text || typeof text !== 'string') return [];
  const raw = text.match(/([A-Za-z']+|[0-9]+)|([^A-Za-z'0-9]+)/g) || [];
  const tokens = [], wordIdx = [];
  for (const chunk of raw) {
    if (/[A-Za-z'0-9]/.test(chunk)) {
      const clean = chunk.toLowerCase().replace(/'/g, '');
      const role = wordRole(clean);
      tokens.push({ text: chunk, word: clean, role, ws: '', emotion: emotionFor(clean, role) });
      wordIdx.push(tokens.length - 1);
    } else if (tokens.length && tokens[tokens.length-1].word !== null && tokens[tokens.length-1].ws === '') {
      tokens[tokens.length-1].ws = chunk;
    } else {
      tokens.push({ text: chunk, word: null, role: 'default', ws: '', emotion: null });
    }
  }
  for (let i = 0; i < wordIdx.length - 1; i++) {
    const a = tokens[wordIdx[i]], b = tokens[wordIdx[i+1]];
    for (const [pair, role] of PHRASES)
      if (a.word === pair[0] && b.word === pair[1]) { a.role = role; b.role = role; a.emotion = b.emotion = null; }
  }
  for (const t of tokens) if (t.word !== null) t.conf = hashConf(t.word, t.role);
  return tokens;
}

export function summarize(tokens) {
  let pos = 0, neg = 0;
  for (const t of tokens) {
    if (t.role === 'emotion-pos') pos += t.conf || 0.7;
    if (t.role === 'emotion-neg') neg += t.conf || 0.7;
  }
  let tone = 'neutral';
  if (pos > neg + 0.4) tone = 'warm';
  else if (neg > pos + 0.4) tone = 'tense';
  else if (pos + neg > 0.4) tone = 'mixed';
  return { tone, pos, neg };
}

export const THEMES = {
  prism: { label:'Prism', blurb:'Full-spectrum, vivid Apple hues', ink:['#3a3a52','#c9cbe8'], roles:{
    'emotion-pos':['#1f9d57','#54e08a'],'emotion-neg':['#d62f3c','#ff6b73'],'question':['#0a84c4','#3fc4ff'],
    'request':['#7a35d6','#b98bff'],'agreement':['#0a9488','#3fe0cf'],'refusal':['#d2541e','#ff8a4c'],
    'hedge':['#9a7b16','#e0c24a'],'intensifier':['#e06a00','#ffa047'],'absolute':['#c01e6f','#ff5fa8'],'negation':['#b03a86','#ff7ed0'] } },
  solar: { label:'Solar', blurb:'Warm, paper-soft, solarized', ink:['#5b5347','#cfc4ad'], roles:{
    'emotion-pos':['#6b8f1e','#a7c84e'],'emotion-neg':['#bf3b2f','#e57a6b'],'question':['#2d7d8f','#73c3d0'],
    'request':['#8a5a9e','#c79bd6'],'agreement':['#3f8a6b','#82c9a6'],'refusal':['#c2651f','#e89b58'],
    'hedge':['#9c7a2c','#d6b665'],'intensifier':['#cf6b25','#f0a25e'],'absolute':['#b1456b','#e088a6'],'negation':['#a15388','#d493bd'] } },
  neon: { label:'Neon', blurb:'Electric, high-contrast, cyberpunk', ink:['#4a4a6a','#aab0e8'], roles:{
    'emotion-pos':['#00a060','#1dff9b'],'emotion-neg':['#e60048','#ff1f6b'],'question':['#0091ff','#27d0ff'],
    'request':['#7b2cff','#c46bff'],'agreement':['#00a896','#1ff0d4'],'refusal':['#ff5a00','#ff8c2e'],
    'hedge':['#b08900','#ffd23f'],'intensifier':['#ff7a00','#ffab38'],'absolute':['#ff0080','#ff4fb0'],'negation':['#d100c9','#ff5ff0'] } },
  graphite: { label:'Graphite', blurb:'Restrained ink, color only where it matters', ink:['#33333f','#c4c6d2'], roles:{
    'emotion-pos':['#2c7d4f','#5fcf8c'],'emotion-neg':['#c0392b','#f0726a'],'question':['#4a5568','#9aa4ba'],
    'request':['#5a4a78','#a99ccb'],'agreement':['#3d6b5a','#7fbfa6'],'refusal':['#a8552e','#d99a73'],
    'hedge':['#6b6450','#b8ad8e'],'intensifier':['#8a5a2c','#caa06b'],'absolute':['#9a3b63','#d18bab'],'negation':['#7a4a6e','#bb96b3'] } },
};

const hexToRgb = (hex) => { const h = (typeof hex === 'string' ? hex : '#000000').replace('#',''); return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)]; };

// roleColor(theme, role, dark) → { color, tint }
export function roleColor(themeName, role, dark) {
  const theme = THEMES[themeName] || THEMES.prism;
  const idx = dark ? 1 : 0;
  if (role === 'default') return { color: theme.ink[idx], tint: 'transparent' };
  const c = (theme.roles[role] || theme.ink)[idx];
  const [r, g, b] = hexToRgb(c);
  return { color: c, tint: `rgba(${r}, ${g}, ${b}, ${dark ? 0.16 : 0.11})` };
}
