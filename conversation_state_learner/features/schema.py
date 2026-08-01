"""Feature schema constants shared across extraction, training, and inference."""

EMOTION_LABELS_28 = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral',
]

BERT_LABELS_7 = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

VADER_KEYS_4 = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']

CDM_CTX_DIM  = 40
MSG_DIM      = 28 + 7 + 4 + CDM_CTX_DIM
WINDOW_SIZE  = 3
WIN_BASE_DIM = WINDOW_SIZE * MSG_DIM
DERIVED_DIM  = 9
WINDOW_DIM   = WIN_BASE_DIM + DERIVED_DIM

N_EMOTIONS = len(EMOTION_LABELS_28)
