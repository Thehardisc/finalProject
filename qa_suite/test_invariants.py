import numpy as np                                             # finite-check on the feature vector
import pytest                                                  # test framework

from shared.constants import EMOTION_LABELS, FEATURE_DIM       # 28 labels + expected FV width
import thresholds as T                                         # invariant thresholds


def test_feature_vector_shape(analyze):                        # FV must be the right shape and finite
    fv = analyze("I'm really happy today")["fv"]               # build a feature vector
    assert fv.shape == (1, FEATURE_DIM), f"feature vector is {fv.shape}, expected (1, {FEATURE_DIM})"  # width
    assert np.all(np.isfinite(fv)), "feature vector contains NaN/Inf"  # no NaN/Inf


def test_goe_gate_capped(analyze, meta_model):                 # GoEmotions gate weight is capped
    r = analyze("I totally love this!")                       # GoE-dominant input
    if r["gate_alpha"] is None:                                # gate only exists for the v2 model
        pytest.skip("gate weights only exposed by GatingEnsembleNet (v2) meta-learner")
    goe_gate = r["gate_alpha"][2]                              # index 2 = GoE gate
    assert goe_gate <= T.GOE_GATE_CAP + 1e-6, (                # must respect the cap
        f"GoEmotions gate {goe_gate:.3f} exceeds documented cap {T.GOE_GATE_CAP}")


def test_gate_vector_shape(analyze):                           # gate vector shape + range
    ga = analyze("This makes me so angry")["gate_alpha"]       # gate weights
    if ga is None:                                             # v2-only
        pytest.skip("gate weights only exposed by GatingEnsembleNet (v2) meta-learner")
    assert len(ga) == T.GATE_VECTOR_LEN, f"gate vector has {len(ga)} elems, expected {T.GATE_VECTOR_LEN}"  # 5 elems
    assert all(0.0 <= float(a) <= 1.0 for a in ga), f"gate weights out of [0,1]: {ga}"  # each in [0,1]


def test_scores_normalised(analyze, meta_model):               # distribution sums to ~1
    if meta_model is None:                                     # fallback returns raw GoE mass, not a distribution
        pytest.skip("rule-based fallback returns raw GoE mass, not a normalised distribution")
    scores = analyze("I am completely devastated")["all_scores"]  # full score dict
    total = sum(scores[k] for k in EMOTION_LABELS if k in scores)  # sum over the 28 labels only
    assert abs(total - 1.0) <= T.SCORE_SUM_TOL, f"scores sum to {total:.3f}, expected ~1.0"  # ~1.0
