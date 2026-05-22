// frontend_service/src/constants/emotions.js
// Shared emotion constants used across views, hooks, and visualizations.
// API_BASE and WS_BASE are now in src/api/client.js (env-aware).

export const EMOTION_COLORS = {
    'joy':        '#00ff88',
    'happy':      '#00ff88',
    'love':       '#ff66b2',
    'admiration': '#00ffcc',
    'anger':      '#ff0055',
    'annoyance':  '#ff6600',
    'disgust':    '#ff00aa',
    'sadness':    '#0066ff',
    'remorse':    '#5500ff',
    'fear':       '#7000ff',
    'nervousness':'#9900ff',
    'surprise':   '#ff9900',
    'curiosity':  '#ffcc00',
    'neutral':    '#00f2ff',
};

export const EMOTIONAL_WORDS = {
    'love':   'joy',     'happy':  'joy',    'great': 'joy', 'joy': 'joy', 'fun': 'joy',
    'hate':   'anger',   'angry':  'anger',  'mad':   'anger',
    'stupid': 'anger',   'hell':   'anger',  'kill':  'anger',
    'sad':    'sadness', 'cry':    'sadness', 'grief': 'sadness',
    'sorry':  'sadness', 'regret': 'sadness',
    'fear':   'fear',    'scared': 'fear',   'afraid':'fear', 'worry': 'fear',
    'wow':    'surprise','shock':  'surprise','amazing':'surprise',
};
