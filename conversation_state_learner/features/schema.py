"""
Feature schema constants shared across extraction, training, and inference.

Feature vector per message — 77 dims total:
  [  0: 27]  go_emotions    — 28 GoEmotions softmax scores
  [ 28: 34]  basic_bert     — 7 basic emotion scores
  [ 35: 38]  vader          — neg, neu, pos, compound
  [ 39: 76]  context_engine — 38 dims context vector

Window feature vector — WIN_SIZE messages × 77 + 9 derived = 240 dims (WIN_SIZE=3):
  [  0:230]  flattened per-message features (3 × 77 = 231)
  [231]      valence_slope           — linear regression slope of compound over window
  [232]      valence_variance        — variance of compound over window
  [233:235]  model_agreement_t0..t2  — go_emotions vs bert top-emotion agreement per msg
  [236:238]  entropy_t0..t2          — Shannon entropy of go_emotions distribution per msg
  [239]      emotion_shift_count     — number of dominant-emotion changes in window

LSTM sequence input per timestep — 77 dims (same as per-message above)
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
MSG_DIM    = 28 + 7 + 4 + 38    # 77 per message
WINDOW_SIZE = 3
WIN_BASE_DIM = WINDOW_SIZE * MSG_DIM    # 231
DERIVED_DIM  = 9
WINDOW_DIM   = WIN_BASE_DIM + DERIVED_DIM  # 240

N_EMOTIONS = len(EMOTION_LABELS_28)    # 28 — target dimension
