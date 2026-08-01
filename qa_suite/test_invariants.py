import numpy as np
import pytest

from shared.constants import EMOTION_LABELS, FEATURE_DIM
import thresholds as T


def test_feature_vector_shape(analyze):
    fv = analyze("I'm really happy today")["fv"]
    assert fv.shape == (1, FEATURE_DIM), f"feature vector is {fv.shape}, expected (1, {FEATURE_DIM})"
    assert np.all(np.isfinite(fv)), "feature vector contains NaN/Inf"


def test_goe_gate_capped(analyze, meta_model):
    r = analyze("I totally love this!")
    if r["gate_alpha"] is None:
        pytest.skip("gate weights only exposed by GatingEnsembleNet (v2) meta-learner")
    goe_gate = r["gate_alpha"][2]
    assert goe_gate <= T.GOE_GATE_CAP + 1e-6, (
        f"GoEmotions gate {goe_gate:.3f} exceeds documented cap {T.GOE_GATE_CAP}")


def test_gate_vector_shape(analyze):
    ga = analyze("This makes me so angry")["gate_alpha"]
    if ga is None:
        pytest.skip("gate weights only exposed by GatingEnsembleNet (v2) meta-learner")
    assert len(ga) == T.GATE_VECTOR_LEN, f"gate vector has {len(ga)} elems, expected {T.GATE_VECTOR_LEN}"
    assert all(0.0 <= float(a) <= 1.0 for a in ga), f"gate weights out of [0,1]: {ga}"


def test_scores_normalised(analyze, meta_model):
    if meta_model is None:
        pytest.skip("rule-based fallback returns raw GoE mass, not a normalised distribution")
    scores = analyze("I am completely devastated")["all_scores"]
    total = sum(scores[k] for k in EMOTION_LABELS if k in scores)
    assert abs(total - 1.0) <= T.SCORE_SUM_TOL, f"scores sum to {total:.3f}, expected ~1.0"
