# Gates below were recalibrated 2026-07-21 against the shipped meta_weights.pkl, each set to
# the measured baseline minus a safety margin (tight enough to catch a real regression, loose
# enough to survive normal retrain noise). Baselines were captured via qa_suite/calibrate.py and
# full offline+slow+live runs — see the comment above each gate for the measured number.

CLEAR_CONF_FLOOR = 0.83  # calibrate.py: min passing final_conf=0.882 on SENTENCE_BATTERY (12/12), margin 0.05
MIXED_FLOOR = 0.10  # measured weakest qualifying secondary signal 0.138 ("hilarious...infuriating" -> annoyance), margin 0.038
VAGUE_ENTROPY_MIN = 0.55  # already above the clear-input entropy ceiling (0.42) — no change needed
VAGUE_CONF_MAX = 0.55

NEGATION_MIN_DELTA = 0.5      # measured delta 0.782 ("this is good" +0.440 vs "this is not good" -0.341)
INTENSIFIER_MIN_RATIO = 1.02  # measured ratio 1.069 ("I am happy" +0.572 vs "I am very happy" +0.612)
EMOJI_MIN_DELTA = 0.25        # measured min |delta| 0.361 (😀 +0.361, 😢 -0.477, vs "lunch" baseline 0.000)

GOE_GATE_CAP = 0.50       # architectural clamp (GatingEnsembleNet), not a QA calibration target
GATE_VECTOR_LEN = 5       # structural: [vader, bert, goe, vad, ctx]
SCORE_SUM_TOL = 0.02

CORPUS_ACCURACY_GATE = 0.88   # measured 0.938 (210/224) on the 28-emotion coverage corpus
FAMILY_ACCURACY_FLOOR = 0.75  # measured per-family: joy .969 anger .875 sadness .950 fear .938
                              # surprise .875 neutral 1.00 — floor sits below the weakest (anger/surprise .875)

FUZZ_N = 3000  # measured 3000/3000 (100%) no-crash/valid-label — inherently a must-never-fail gate

EDGE_FAMILY_GATE = 0.62   # measured 0.688 (75/109) on the curated hard-edge-case corpus

GOEMOTIONS_FAMILY_GATE = 0.65  # measured 0.715 family-acc (n=2000 real GoEmotions, top1-exact 0.613)

CONV_FAMILY_GATE = 0.68  # measured 0.755 (40/53 turns) on the 12 curated conversations
CONV_TREND_EPS = 0.05

CONV_GEN_N = 3000
CONV_GEN_FAMILY_GATE = 0.88  # measured 0.961 per-turn family-acc (13,250 turns, 3000 generated convs)
CONV_ARC_GATE = 0.80         # measured 0.845 (2536/3000) arc-trajectory match; kept slack — known
                              # weak arcs (celebration/good_news/disgust/gratitude) carry most misses

PERF_LATENCY_BUDGET_SEC = 3.0  # measured warm latency 0.26-0.36s; ~9x headroom for docker/network jitter
BURST_N = 20
SOAK_REGRESSION_FACTOR = 1.5  # measured 2nd-half/1st-half ratio ~1.1x under sustained load; still
                              # catches a real 50%+ degradation without flagging normal jitter

PERSIST_WAIT_SEC = 20
ANALYSIS_WAIT_SEC = 15
ANALYSIS_LATENCY_BUDGET_SEC = 5.0  # measured 0.01s (analyze is a fast Redis+DB round trip); generous
                                   # headroom since /analyze cost scales with conversation length
