# shared/constants.py

# Complete list of 28 GoEmotions labels (including Neutral)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Fixed ordering for the Meta-Learner feature vector
VADER_KEYS  = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
BERT_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Feature Dimension: VADER(4) + BERT(7) + GoEmotions(28) + EmojiNet(28)
#                    + Valence(1) + OneHotPrevMood(28) + Derived(7) = 103
FEATURE_DIM = 103

# emojinet knowledge base
# Single source of truth for emoji → emotion mappings.
# Used by both the inline _emojinet() call during training AND inference.
# Only the 'emotions' dict is used for feature vectors; keep entries consistent.
EMOJI_EMOTION_DB: dict[str, dict] = {
    "😂": {"emotions": {"joy": 0.9,  "amusement": 0.95},                              "sarcasm_potential": 0.4},
    "😭": {"emotions": {"sadness": 0.8, "grief": 0.6,   "joy": 0.2},                  "sarcasm_potential": 0.6},
    "😍": {"emotions": {"love": 0.95, "admiration": 0.9, "joy": 0.8},                 "sarcasm_potential": 0.1},
    "🔥": {"emotions": {"excitement": 0.9, "admiration": 0.8, "joy": 0.7},           "sarcasm_potential": 0.2},
    "💀": {"emotions": {"amusement": 0.9, "joy": 0.7,  "fear": 0.1},                 "sarcasm_potential": 0.8},
    "🙃": {"emotions": {"amusement": 0.4, "annoyance": 0.6, "confusion": 0.3},       "sarcasm_potential": 0.95},
    "🤔": {"emotions": {"curiosity": 0.8, "confusion": 0.4, "disapproval": 0.2},     "sarcasm_potential": 0.5},
    "🙄": {"emotions": {"annoyance": 0.9, "disapproval": 0.8, "disgust": 0.5},       "sarcasm_potential": 0.9},
    "💩": {"emotions": {"disgust": 0.8, "amusement": 0.5, "annoyance": 0.4},         "sarcasm_potential": 0.3},
    "❤️":  {"emotions": {"love": 1.0,  "caring": 0.9,  "joy": 0.8},                  "sarcasm_potential": 0.05},
    "✨":   {"emotions": {"excitement": 0.7, "admiration": 0.6, "joy": 0.5},           "sarcasm_potential": 0.7},
    "😊": {"emotions": {"joy": 0.85, "approval": 0.7,  "caring": 0.5},               "sarcasm_potential": 0.1},
    "😔": {"emotions": {"sadness": 0.85, "disappointment": 0.7, "remorse": 0.4},     "sarcasm_potential": 0.1},
    "😡": {"emotions": {"anger": 0.95, "annoyance": 0.8, "disapproval": 0.6},        "sarcasm_potential": 0.2},
    "🥺": {"emotions": {"desire": 0.7, "sadness": 0.5,  "caring": 0.4},              "sarcasm_potential": 0.15},
}
