"""
Feature-vector sanity test for the 108-dim architecture.

Layout: VADER(4) + BERT(7) + GoE(28) + CDM_CTX(40) + TRAJ_PRIOR(28) + SARCASM(1) = 108

Both inference (meta_learner.py:build_feature_vector) and training
share the same build_feature_vector from meta_learner.py, so this test verifies:
  1. The function produces a (1, FEATURE_DIM) array with the right shape.
  2. VADER / BERT / GoEmotions values land in the correct index blocks.
  3. CDM context and trajectory prior default to zeros when missing.
  4. sarcasm_score lands at index [107] and is clipped to [0, 1].
  5. FEATURE_DIM, CONTEXT_DIM, CDM_CTX_DIM, PRIOR_DIM, SARCASM_DIM match constants.

Run from project root:
    python -m pytest central_responder_service/training/test_feature_parity.py -v
"""
import os
import sys
import numpy as np
import pytest

_HERE         = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SVC_ROOT     = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_PROJECT_ROOT, _SVC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from meta_learner import build_feature_vector
from shared.constants import (
    EMOTION_LABELS, BERT_LABELS, VADER_KEYS,
    FEATURE_DIM, ML_DIM, CONTEXT_DIM, CDM_CTX_DIM, PRIOR_DIM, SARCASM_DIM,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _model_outputs(neg=0.0, neu=0.0, pos=0.0, compound=0.0, bert=None, goe=None):
    return {
        "vader": {
            "vader_neg": neg, "vader_neu": neu,
            "vader_pos": pos, "vader_compound": compound,
        },
        "basic_bert":  bert or {k: 0.0 for k in BERT_LABELS},
        "go_emotions": goe  or {k: 0.0 for k in EMOTION_LABELS},
    }


def _ctx(length=CDM_CTX_DIM, value=0.0):
    return [value] * length

def _prior(length=PRIOR_DIM, value=0.0):
    return [value] * length


# ── Constant layout checks ────────────────────────────────────────────────────

def test_feature_dim_is_108():
    assert FEATURE_DIM == 108

def test_sarcasm_dim_is_1():
    assert SARCASM_DIM == 1

def test_ml_dim_is_39():
    assert ML_DIM == 39

def test_cdm_ctx_dim_is_40():
    assert CDM_CTX_DIM == 40

def test_prior_dim_is_28():
    assert PRIOR_DIM == 28

def test_context_dim_is_69():
    assert CONTEXT_DIM == 69

def test_ml_plus_context_equals_feature():
    assert ML_DIM + CONTEXT_DIM == FEATURE_DIM

def test_label_lengths():
    assert len(VADER_KEYS)     == 4
    assert len(BERT_LABELS)    == 7
    assert len(EMOTION_LABELS) == 28


# ── Shape checks ──────────────────────────────────────────────────────────────

def test_output_shape_no_context():
    fv = build_feature_vector(_model_outputs())
    assert fv.shape == (1, FEATURE_DIM)

def test_output_shape_with_context():
    fv = build_feature_vector(_model_outputs(), context_vector=_ctx(), trajectory_prior=_prior())
    assert fv.shape == (1, FEATURE_DIM)

def test_output_dtype_is_float32():
    fv = build_feature_vector(_model_outputs())
    assert fv.dtype == np.float32


# ── Block placement checks ────────────────────────────────────────────────────

def test_vader_block():
    mo = _model_outputs(neg=0.1, neu=0.2, pos=0.3, compound=0.4)
    fv = build_feature_vector(mo).flatten()
    expected = [
        mo["vader"][k] for k in VADER_KEYS
    ]
    np.testing.assert_allclose(fv[0:4], expected, atol=1e-6)

def test_bert_block():
    bert = {k: float(i) / 10 for i, k in enumerate(BERT_LABELS)}
    mo = _model_outputs(bert=bert)
    fv = build_feature_vector(mo).flatten()
    expected = [bert[k] for k in BERT_LABELS]
    np.testing.assert_allclose(fv[4:11], expected, atol=1e-6)

def test_goemotions_block():
    goe = {k: float(i) / 100 for i, k in enumerate(EMOTION_LABELS)}
    mo = _model_outputs(goe=goe)
    fv = build_feature_vector(mo).flatten()
    expected = [goe[k] for k in EMOTION_LABELS]
    np.testing.assert_allclose(fv[11:39], expected, atol=1e-6)

def test_cdm_block_zeros_when_missing():
    fv = build_feature_vector(_model_outputs(), context_vector=None).flatten()
    np.testing.assert_array_equal(fv[39:79], [0.0] * CDM_CTX_DIM)

def test_cdm_block_zeros_when_wrong_length():
    fv = build_feature_vector(_model_outputs(), context_vector=[0.5] * 10).flatten()
    np.testing.assert_array_equal(fv[39:79], [0.0] * CDM_CTX_DIM)

def test_cdm_block_used_when_correct_length():
    from shared.constants import CTX_MSG_LENGTH, CTX_LATENCY_MS, CTX_HMM_EMISSION
    ctx = [float(i) / 100 for i in range(CDM_CTX_DIM)]
    fv = build_feature_vector(_model_outputs(), context_vector=ctx).flatten()
    cdm_out = fv[39:79]
    # indices 31, 32, 35 are normalized in build_feature_vector — skip exact match there
    normalized_idx = {CTX_MSG_LENGTH, CTX_LATENCY_MS, CTX_HMM_EMISSION}
    for i, (actual, expected) in enumerate(zip(cdm_out, ctx)):
        if i not in normalized_idx:
            assert actual == pytest.approx(expected, abs=1e-6), f"CDM index {i} mismatch"
    # confirm normalized slots are present (non-NaN) and within their expected ranges
    assert 0.0 <= cdm_out[CTX_MSG_LENGTH]  <= 3.0   # raw_chars / 500, capped at 3
    assert 0.0 <= cdm_out[CTX_LATENCY_MS]  <= 3.0   # raw_ms / 2000, capped at 3
    assert 0.0 <= cdm_out[CTX_HMM_EMISSION] <= 1.0  # already [0,1] from context_engine, clipped

def test_prior_block_zeros_when_missing():
    fv = build_feature_vector(_model_outputs(), trajectory_prior=None).flatten()
    np.testing.assert_array_equal(fv[79:107], [0.0] * PRIOR_DIM)

def test_prior_block_used_when_correct_length():
    prior = [float(i) / 100 for i in range(PRIOR_DIM)]
    fv = build_feature_vector(_model_outputs(), trajectory_prior=prior).flatten()
    np.testing.assert_allclose(fv[79:107], prior, atol=1e-6)


# ── Sarcasm slot [107] ────────────────────────────────────────────────────────

def test_sarcasm_slot_default_zero():
    fv = build_feature_vector(_model_outputs()).flatten()
    assert fv[107] == pytest.approx(0.0, abs=1e-6)

def test_sarcasm_slot_value_stored():
    fv = build_feature_vector(_model_outputs(), sarcasm_score=0.83).flatten()
    assert fv[107] == pytest.approx(0.83, abs=1e-5)

def test_sarcasm_slot_clipped_above_one():
    fv = build_feature_vector(_model_outputs(), sarcasm_score=1.5).flatten()
    assert fv[107] == pytest.approx(1.0, abs=1e-5)

def test_sarcasm_slot_clipped_below_zero():
    fv = build_feature_vector(_model_outputs(), sarcasm_score=-0.3).flatten()
    assert fv[107] == pytest.approx(0.0, abs=1e-5)

def test_sarcasm_does_not_affect_prior_block():
    prior = [float(i) / 100 for i in range(PRIOR_DIM)]
    fv = build_feature_vector(_model_outputs(), trajectory_prior=prior, sarcasm_score=0.7).flatten()
    np.testing.assert_allclose(fv[79:107], prior, atol=1e-6)
    assert fv[107] == pytest.approx(0.7, abs=1e-5)


# ── Missing keys default to zero ──────────────────────────────────────────────

def test_missing_model_keys_default_to_zero():
    # All model outputs missing, no sarcasm_score → entire vector should be zeros.
    fv = build_feature_vector({}).flatten()
    assert fv.shape == (FEATURE_DIM,)
    np.testing.assert_array_equal(fv, [0.0] * FEATURE_DIM)

def test_partial_bert_keys_default_missing_to_zero():
    mo = _model_outputs(bert={"joy": 0.9})
    fv = build_feature_vector(mo).flatten()
    joy_idx = 4 + BERT_LABELS.index("joy")
    assert fv[joy_idx] == pytest.approx(0.9, abs=1e-6)
    # all other BERT positions should be 0
    for i, k in enumerate(BERT_LABELS):
        if k != "joy":
            assert fv[4 + i] == pytest.approx(0.0, abs=1e-6)
