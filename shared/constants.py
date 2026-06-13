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

# ── CDM v4.0 — 15 intent-based conversational states ─────────────────────────
# Index order MUST match cdm.py state constants and CDMStateGraph.jsx.
CDM_STATES = [
    "NEUTRAL",         # 0  — no clear intent
    "WARMTH",          # 1  — expressing care/kindness
    "PRAISE",          # 2  — complimenting or approving
    "HELP_REQUEST",    # 3  — asking for assistance
    "HUMOR",           # 4  — playful or funny
    "TENSION",         # 5  — informing with frustration
    "CONFLICT",        # 6  — confrontational directive
    "ARGUMENT",        # 7  — challenging question
    "WITHDRAWAL",      # 8  — pulling back emotionally
    "RECONCILIATION",  # 9  — repairing after conflict
    "CURIOSITY",       # 10 — exploring or inquiring
    "ASSERTIVENESS",   # 11 — stating position firmly
    "EMPATHY",         # 12 — responding with care to distress
    "FRUSTRATION",     # 13 — frustrated questioning
    "AGREEMENT",       # 14 — expressing alignment
]
N_CDM_STATES = len(CDM_STATES)  # 15

# ── Feature vector layout ─────────────────────────────────────────────────────
#
#   ML_DIM         [0:39]    = VADER(4) + BERT(7) + GoEmotions(28)
#   CDM_CTX_DIM    [39:79]   = CDM one-hot(15) + 16 scalars + 7 HMM-derived
#   PRIOR_DIM      [79:107]  = trajectory LSTM predicted_next (28 GoEmotions labels)
#
# Context vector internal layout (38 dims):
#   [0:15]   CDM intent one-hot (15 states — exactly one 1.0)
#   [15]     state_residency         (how long in current state, 0-1)
#   [16:19]  transition_path         (last 3 state indices / N_CDM_STATES)
#   [19]     entry_abruptness        (1 - transition_probability)
#   [20]     topic_coherence         (cosine similarity with previous embedding)
#   [21]     emotion_entropy         (Shannon entropy of GoEmotions distribution)
#   [22]     speaker_divergence      (std of all speaker valences in conversation)
#   [23]     velocity                (Δvalence between last 2 messages)
#   [24]     acceleration            (Δ²valence — is shift intensifying or stabilising?)
#   [25]     historical_pos          (Qdrant positive memory intensity)
#   [26]     historical_neu          (Qdrant neutral memory intensity)
#   [27]     historical_neg          (Qdrant negative memory intensity)
#   [28]     topic_resonance         (Qdrant similarity score)
#   [29]     volatility              (EMA of variance)
#   [30]     current_valence         (EMA valence of conversation)
#   [31]     message_length
#   [32]     latency_ms
#   [33]     hmm_posterior_confidence (max of forward variable α)
#   [34]     hmm_state_entropy        (H(α))
#   [35]     hmm_emission_logprob     (log P(obs|state), normalised to [0,1])
#   [36:39]  hmm_top3_next_probs      (top-3 P(s_{t+1}|s_t) from learned transmat)
#   [39]     intent_stability         (consecutive messages in same intent / 10)

ML_DIM      = 39    # VADER(4) + BERT(7) + GoE(28)
CDM_CTX_DIM = 40    # context_engine_service output: CDM one-hot(15) + 18 scalars + 7 HMM
PRIOR_DIM   = 28    # trajectory LSTM predicted_next distribution (one per GoEmotions label)
SARCASM_DIM = 1     # learned sarcasm/passive-aggression score from sarcasm_classifier.py
CONTEXT_DIM = CDM_CTX_DIM + PRIOR_DIM + SARCASM_DIM  # = 69
FEATURE_DIM = ML_DIM + CONTEXT_DIM                   # = 108

# Named index map — import these instead of hardcoding integers.
# Any layout change must update both these constants AND context_engine_service/main.py.
CTX_CDM_PROBS      = slice(0, N_CDM_STATES)                      # [0:15]
CTX_RESIDENCY      = N_CDM_STATES                                 # 15
CTX_TRANSITION     = slice(N_CDM_STATES + 1, N_CDM_STATES + 4)   # [16:19]
CTX_ABRUPTNESS     = N_CDM_STATES + 4                             # 19
CTX_COHERENCE      = N_CDM_STATES + 5                             # 20
CTX_ENTROPY        = N_CDM_STATES + 6                             # 21
CTX_SPK_DIVERGENCE = N_CDM_STATES + 7                             # 22
CTX_VELOCITY       = N_CDM_STATES + 8                             # 23
CTX_ACCELERATION   = N_CDM_STATES + 9                             # 24
CTX_HIST_POS       = N_CDM_STATES + 10                            # 25
CTX_HIST_NEU       = N_CDM_STATES + 11                            # 26
CTX_HIST_NEG       = N_CDM_STATES + 12                            # 27
CTX_RESONANCE      = N_CDM_STATES + 13                            # 28
CTX_VOLATILITY     = N_CDM_STATES + 14                            # 29
CTX_CURR_VALENCE   = N_CDM_STATES + 15                            # 30
CTX_MSG_LENGTH     = N_CDM_STATES + 16                            # 31
CTX_LATENCY_MS     = N_CDM_STATES + 17                            # 32
CTX_HMM_CONF       = N_CDM_STATES + 18                            # 33
CTX_HMM_ENTROPY    = N_CDM_STATES + 19                            # 34
CTX_HMM_EMISSION   = N_CDM_STATES + 20                            # 35
CTX_HMM_NEXT3      = slice(N_CDM_STATES + 21, N_CDM_STATES + 24) # [36:39]
CTX_INTENT_STAB    = N_CDM_STATES + 24                            # 39
