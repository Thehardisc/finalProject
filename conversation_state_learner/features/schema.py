"""
Feature schema constants shared across extraction, training, and inference.

Feature vector per message — 79 dims total:
  [  0: 27]  go_emotions    — 28 GoEmotions softmax scores
  [ 28: 34]  basic_bert     — 7 basic emotion scores
  [ 35: 38]  vader          — neg, neu, pos, compound
  [ 39: 78]  context_engine — 40 dims CDM context vector (CDM_CTX_DIM=40)

Window feature vector — WIN_SIZE messages × 79 + 9 derived = 246 dims (WIN_SIZE=3):
  [  0:236]  flattened per-message features (3 × 79 = 237)
  [237]      valence_slope           — linear regression slope of compound over window
  [238]      valence_variance        — variance of compound over window
  [239:241]  model_agreement_t0..t2  — go_emotions vs bert top-emotion agreement per msg
  [242:244]  entropy_t0..t2          — Shannon entropy of go_emotions distribution per msg
  [245]      emotion_shift_count     — number of dominant-emotion changes in window

LSTM sequence input per timestep — 79 dims (same as per-message above)
LSTM target per timestep        — 28 dims (go_emotions distribution of NEXT message)

CDM_CTX_DIM = 40 (matches shared/constants.py — was 38 before the CDM v4 HMM expansion)
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
CDM_CTX_DIM  = 40               # must match shared/constants.py CDM_CTX_DIM
MSG_DIM      = 28 + 7 + 4 + CDM_CTX_DIM   # 79 per message
WINDOW_SIZE  = 3
WIN_BASE_DIM = WINDOW_SIZE * MSG_DIM       # 237
DERIVED_DIM  = 9
WINDOW_DIM   = WIN_BASE_DIM + DERIVED_DIM  # 246

N_EMOTIONS = len(EMOTION_LABELS_28)    # 28 — target dimension
