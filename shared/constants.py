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

# Feature Dimension: VADER(4) + BERT(7) + GoEmotions(28) + EmojiNet(28) + Valence(1) + OneHotPrevMood(28) = 96
FEATURE_DIM = 96
