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

# ─────────────────────────────────────────────────────────────────────────────
# Feature Vector layout — single source of truth (FEATURE_DIM = 118)
#
#   VADER(4) + BERT(7) + GoEmotions(28) + EmojiNet(28)            = 67  (model block)
#   + CDM Context(44)                                             = 111
#   + Derived(7)                                                  = 118
#
# The CDM Context block (44) replaces the legacy "avg_valence + one-hot prev
# emotion" (29) block.  It is produced by context_engine_service's Parallel
# Belief State Machine (see context_engine_service/cdm.py).
# ─────────────────────────────────────────────────────────────────────────────

# Model-output block boundaries within the full feature vector
FV_VADER_START   = 0
FV_BERT_START    = 4
FV_GOE_START     = 11
FV_EMOJI_START   = 39
FV_CONTEXT_START = 67     # CDM context block begins here
FV_DERIVED_START = 111    # derived features begin here

ML_DIM      = 67    # everything before the context block (VADER+BERT+GoE+Emoji)
CONTEXT_DIM = 44    # 28-dim emotion belief + 16 CDM scalars
DERIVED_DIM = 7
FEATURE_DIM = ML_DIM + CONTEXT_DIM + DERIVED_DIM   # 118

# ─── CDM context-vector layout (indices are RELATIVE to the 44-dim context block) ───
# Any change here must update context_engine_service/main.py AND the feature
# builders in central_responder_service (ml/features.py + trainer/preprocessor.py).
CTX_CDM_PROBS      = slice(0, 28)   # CDM emotion belief distribution (PBSM)
CTX_RESIDENCY      = 28             # how long top-1 state held (0-1)
CTX_TRANSITION     = slice(29, 32)  # last 3 top-1 state indices / 27
CTX_ABRUPTNESS     = 32             # entry abruptness (0-1)
CTX_COHERENCE      = 33             # topic coherence (cosine similarity)
CTX_ENTROPY        = 34             # Shannon entropy of GoEmotions distribution
CTX_SPK_DIVERGENCE = 35             # speaker divergence (std of per-speaker valences)
CTX_VELOCITY       = 36             # Δvalence vs previous message
CTX_ACCELERATION   = 37             # Δ²valence
CTX_HIST_VALENCE   = 38             # Qdrant episodic memory valence
CTX_RESONANCE      = 39             # Qdrant topic resonance score
CTX_VOLATILITY     = 40             # EMA variance
CTX_CURR_VALENCE   = 41             # EMA valence of conversation
CTX_MSG_LENGTH     = 42             # character count
CTX_LATENCY_MS     = 43             # time since previous message

# Number of CDM belief states (one per GoEmotions label)
CDM_N_STATES = len(EMOTION_LABELS)   # 28
N_CDM_STATES = CDM_N_STATES          # backward-compat alias
