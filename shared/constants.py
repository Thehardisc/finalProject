EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

VADER_KEYS  = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
BERT_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

GOE_TO_PLUTCHIK = {
    'joy':         'joy',
    'admiration':  'trust',
    'fear':        'fear',
    'nervousness': 'fear',
    'surprise':    'surprise',
    'sadness':     'sadness',
    'grief':       'sadness',
    'disgust':     'disgust',
    'anger':       'anger',
    'annoyance':   'anger',
    'curiosity':   'anticipation',
}

CDM_STATES = [
    "NEUTRAL",
    "WARMTH",
    "PRAISE",
    "HELP_REQUEST",
    "HUMOR",
    "TENSION",
    "CONFLICT",
    "ARGUMENT",
    "WITHDRAWAL",
    "RECONCILIATION",
    "CURIOSITY",
    "ASSERTIVENESS",
    "EMPATHY",
    "FRUSTRATION",
    "AGREEMENT",
]
N_CDM_STATES = len(CDM_STATES)


ML_DIM        = 42
CDM_CTX_DIM   = 40
PRIOR_DIM     = 28
SARCASM_DIM   = 1
VAD_DIM       = 3
DYNAMICS_DIM  = 2
APPRAISAL_DIM = 3
CONTEXT_DIM   = CDM_CTX_DIM + PRIOR_DIM + SARCASM_DIM + DYNAMICS_DIM + APPRAISAL_DIM
FEATURE_DIM   = ML_DIM + CONTEXT_DIM

CTX_CDM_PROBS      = slice(0, N_CDM_STATES)
CTX_RESIDENCY      = N_CDM_STATES
CTX_TRANSITION     = slice(N_CDM_STATES + 1, N_CDM_STATES + 4)
CTX_ABRUPTNESS     = N_CDM_STATES + 4
CTX_COHERENCE      = N_CDM_STATES + 5
CTX_ENTROPY        = N_CDM_STATES + 6
CTX_SPK_DIVERGENCE = N_CDM_STATES + 7
CTX_VELOCITY       = N_CDM_STATES + 8
CTX_ACCELERATION   = N_CDM_STATES + 9
CTX_HIST_POS       = N_CDM_STATES + 10
CTX_HIST_NEU       = N_CDM_STATES + 11
CTX_HIST_NEG       = N_CDM_STATES + 12
CTX_RESONANCE      = N_CDM_STATES + 13
CTX_VOLATILITY     = N_CDM_STATES + 14
CTX_CURR_VALENCE   = N_CDM_STATES + 15
CTX_MSG_LENGTH     = N_CDM_STATES + 16
CTX_LATENCY_MS     = N_CDM_STATES + 17
CTX_HMM_CONF       = N_CDM_STATES + 18
CTX_HMM_ENTROPY    = N_CDM_STATES + 19
CTX_HMM_EMISSION   = N_CDM_STATES + 20
CTX_HMM_NEXT3      = slice(N_CDM_STATES + 21, N_CDM_STATES + 24)
CTX_INTENT_STAB    = N_CDM_STATES + 24
