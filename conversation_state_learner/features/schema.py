"""
Feature schema constants shared across extraction, training, and inference.

Feature vector per message — 67 dims total:
  [  0: 27]  go_emotions    — 28 GoEmotions softmax scores
  [ 28: 34]  basic_bert     — 7 basic emotion scores
  [ 35: 38]  vader          — neg, neu, pos, compound
  [ 39: 66]  emojinet       — 28 dims aligned to GoEmotions label space (0 if no emoji)

Window feature vector — WIN_SIZE messages × 67 + 9 derived = 210 dims (WIN_SIZE=3):
  [  0:200]  flattened per-message features (3 × 67 = 201)
  [201]      valence_slope           — linear regression slope of compound over window
  [202]      valence_variance        — variance of compound over window
  [203:205]  model_agreement_t0..t2  — go_emotions vs bert top-emotion agreement per msg
  [206:208]  entropy_t0..t2          — Shannon entropy of go_emotions distribution per msg
  [209]      emotion_shift_count     — number of dominant-emotion changes in window

LSTM sequence input per timestep — 67 dims (same as per-message above)
LSTM target per timestep        — 28 dims (go_emotions distribution of NEXT message)
"""

EMOTION_LABELS_28 = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral',
]

BERT_LABELS_7 = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

VADER_KEYS_4 = ['neg', 'neu', 'pos', 'compound']

# Derived feature positions in window vector
MSG_DIM    = 28 + 7 + 4 + 28   # 67 per message
WINDOW_SIZE = 3
WIN_BASE_DIM = WINDOW_SIZE * MSG_DIM    # 201
DERIVED_DIM  = 9
WINDOW_DIM   = WIN_BASE_DIM + DERIVED_DIM  # 210

N_EMOTIONS = len(EMOTION_LABELS_28)    # 28 — target dimension
