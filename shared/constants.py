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

# Feature vector layout:
#   ML_DIM  [0:39]  = VADER(4) + BERT(7) + GoEmotions(28)
#   CONTEXT [39:62] = Context Engine 23-dim CDM scalars
#
# Context vector internal layout (23 dims):
#   [0:7]    CDM state one-hot vector (DFSM — exactly one 1.0)
#   [7]      state_residency         (how long in current state, 0-1)
#   [8:11]   transition_path         (last 3 state indices / 7)
#   [11]     entry_abruptness        (how abrupt the state transition was)
#   [12]     topic_coherence         (cosine similarity with previous embedding)
#   [13]     emotion_entropy         (Shannon entropy of GoEmotions distribution)
#   [14]     speaker_divergence      (std of all speaker valences in conversation)
#   [15]     velocity                (Δvalence between last 2 messages)
#   [16]     acceleration            (Δ²valence — is shift intensifying or stabilising?)
#   [17]     historical_valence      (Qdrant episodic memory)
#   [18]     topic_resonance         (Qdrant similarity score)
#   [19]     volatility              (EMA of variance)
#   [20]     current_valence         (EMA valence of conversation)
#   [21]     message_length
#   [22]     latency_ms
ML_DIM      = 39    # VADER(4) + BERT(7) + GoE(28)
CONTEXT_DIM = 23    # CDM scalars only (no raw embedding)
FEATURE_DIM = 62    # ML_DIM(39) + CONTEXT_DIM(23)

# Named index map for the context vector — import these instead of hardcoding integers.
# Any change to CONTEXT_DIM layout must update these constants AND context_engine_service/main.py.
CTX_CDM_PROBS      = slice(0, 7)   # CDM state one-hot vector (DFSM)
CTX_RESIDENCY      = 7             # how long in current state (0-1)
CTX_TRANSITION     = slice(8, 11)  # last 3 state indices / 7
CTX_ABRUPTNESS     = 11            # entry abruptness (0-1)
CTX_COHERENCE      = 12            # topic coherence (cosine similarity)
CTX_ENTROPY        = 13            # Shannon entropy of GoEmotions distribution
CTX_SPK_DIVERGENCE = 14            # speaker divergence (std of per-speaker valences)
CTX_VELOCITY       = 15            # Δvalence vs previous message
CTX_ACCELERATION   = 16            # Δ²valence
CTX_HIST_VALENCE   = 17            # Qdrant episodic memory valence
CTX_RESONANCE      = 18            # Qdrant topic resonance score
CTX_VOLATILITY     = 19            # EMA variance
CTX_CURR_VALENCE   = 20            # EMA valence of conversation
CTX_MSG_LENGTH     = 21            # character count
CTX_LATENCY_MS     = 22            # time since previous message

# CDM — 7 latent conversation states
CDM_STATES = [
    "NEUTRAL_EXPLORE",    # 0 — low intensity, topic wandering
    "ASCENDING_POSITIVE", # 1 — improving mood, positive momentum
    "CONFLICT_RISE",      # 2 — negative velocity, escalating tension
    "TOPIC_ANCHOR",       # 3 — focused topic, stable low-entropy emotion
    "CONVERGENCE",        # 4 — speakers aligning emotionally
    "EMOTIONAL_PEAK",     # 5 — high entropy, chaotic emotion distribution
    "DIVERGING",          # 6 — speakers moving apart emotionally
]
N_CDM_STATES = len(CDM_STATES)
