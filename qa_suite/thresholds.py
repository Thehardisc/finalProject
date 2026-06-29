CLEAR_CONF_FLOOR = 0.20        # min confidence a clear-emotion input must clear (only when a model is loaded)
MIXED_FLOOR = 0.07             # min mass each of the top-2 emotions needs to count as "mixed"
VAGUE_ENTROPY_MIN = 0.55       # normalised GoE entropy above which a vague input counts as "uncertain"
VAGUE_CONF_MAX = 0.55          # confidence below which a vague input counts as "uncertain"

GOE_GATE_CAP = 0.50            # documented hard cap on the GoEmotions gate weight
GATE_VECTOR_LEN = 5            # gate vector is [vader, bert, goe, vad, ctx]
SCORE_SUM_TOL = 0.02           # tolerance for the 28 scores summing to ~1.0

CORPUS_ACCURACY_GATE = 0.80    # global family-accuracy floor for the curated per-emotion corpus
FAMILY_ACCURACY_FLOOR = 0.65   # per-family floor for the curated per-emotion corpus

FUZZ_N = 3000                  # number of fuzz/robustness inputs

EDGE_FAMILY_GATE = 0.55        # aggregate family-accuracy floor over all authored edge messages

GOEMOTIONS_FAMILY_GATE = 0.55  # family-accuracy floor on real GoEmotions (passes in both model + fallback modes)

CONV_FAMILY_GATE = 0.60        # per-turn family-accuracy floor for the curated conversations
CONV_TREND_EPS = 0.05          # min VADER-compound delta to register an up/down valence move

CONV_GEN_N = 3000              # number of generated full conversations
CONV_GEN_FAMILY_GATE = 0.70    # per-turn family-accuracy floor over generated conversations
CONV_ARC_GATE = 0.75           # fraction of generated convs whose trajectory must match its arc

PERF_LATENCY_BUDGET_SEC = 15.0 # soft budget for a warm POST -> emotion_stream round trip
BURST_N = 20                   # messages in the light throughput probe
