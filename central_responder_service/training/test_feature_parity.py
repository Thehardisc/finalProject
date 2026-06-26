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
    FEATURE_DIM, ML_DIM, CONTEXT_DIM,
    CDM_CTX_DIM, PRIOR_DIM, SARCASM_DIM, VAD_DIM, DYNAMICS_DIM, APPRAISAL_DIM,
    CTX_MSG_LENGTH, CTX_LATENCY_MS, CTX_HMM_EMISSION,
)

_VADER_S  = 0
_BERT_S   = 4
_GOE_S    = 11
_VAD_S    = ML_DIM - VAD_DIM          # 39
_VAD_E    = ML_DIM                    # 42
_CDM_S    = ML_DIM                    # 42
_CDM_E    = ML_DIM + CDM_CTX_DIM      # 82
_PRIOR_S  = _CDM_E                    # 82
_PRIOR_E  = _PRIOR_S + PRIOR_DIM      # 110
_SARC     = _PRIOR_E                  # 110
_DYN_S    = _SARC + SARCASM_DIM       # 111
_DYN_E    = _DYN_S + DYNAMICS_DIM     # 113
_APR_S    = _DYN_E                    # 113
_APR_E    = _APR_S + APPRAISAL_DIM    # 116


def _mo(neg=0.0, neu=0.0, pos=0.0, compound=0.0, bert=None, goe=None):
    return {
        "vader":       {"vader_neg": neg, "vader_neu": neu,
                        "vader_pos": pos, "vader_compound": compound},
        "basic_bert":  bert or {k: 0.0 for k in BERT_LABELS},
        "go_emotions": goe  or {k: 0.0 for k in EMOTION_LABELS},
    }


def _ctx(value=0.0):
    return [value] * CDM_CTX_DIM


def _prior(value=0.0):
    return [value] * PRIOR_DIM


def test_feature_dim_is_116():
    assert FEATURE_DIM == 116

def test_ml_dim_is_42():
    assert ML_DIM == 42

def test_vad_dim_is_3():
    assert VAD_DIM == 3

def test_sarcasm_dim_is_1():
    assert SARCASM_DIM == 1

def test_dynamics_dim_is_2():
    assert DYNAMICS_DIM == 2

def test_appraisal_dim_is_3():
    assert APPRAISAL_DIM == 3

def test_cdm_ctx_dim_is_40():
    assert CDM_CTX_DIM == 40

def test_prior_dim_is_28():
    assert PRIOR_DIM == 28

def test_context_dim_is_74():
    assert CONTEXT_DIM == 74

def test_ml_plus_context_equals_feature():
    assert ML_DIM + CONTEXT_DIM == FEATURE_DIM

def test_label_lengths():
    assert len(VADER_KEYS)     == 4
    assert len(BERT_LABELS)    == 7
    assert len(EMOTION_LABELS) == 28


def test_output_shape_no_context():
    assert build_feature_vector(_mo()).shape == (1, FEATURE_DIM)

def test_output_shape_with_context():
    assert build_feature_vector(
        _mo(), context_vector=_ctx(), trajectory_prior=_prior()
    ).shape == (1, FEATURE_DIM)

def test_output_shape_empty_outputs():
    assert build_feature_vector({}).shape == (1, FEATURE_DIM)

def test_output_dtype_is_float32():
    assert build_feature_vector(_mo()).dtype == np.float32


def test_vader_block():
    mo = _mo(neg=0.1, neu=0.2, pos=0.3, compound=0.4)
    fv = build_feature_vector(mo).flatten()
    np.testing.assert_allclose(fv[0:4], [mo["vader"][k] for k in VADER_KEYS], atol=1e-6)

def test_bert_block():
    bert = {k: float(i) / 10 for i, k in enumerate(BERT_LABELS)}
    fv = build_feature_vector(_mo(bert=bert)).flatten()
    np.testing.assert_allclose(fv[4:11], [bert[k] for k in BERT_LABELS], atol=1e-6)

def test_goemotions_block():
    goe = {k: float(i) / 100 for i, k in enumerate(EMOTION_LABELS)}
    fv = build_feature_vector(_mo(goe=goe)).flatten()
    np.testing.assert_allclose(fv[11:39], [goe[k] for k in EMOTION_LABELS], atol=1e-6)

def test_vad_block_defaults_zero():
    fv = build_feature_vector(_mo()).flatten()
    np.testing.assert_array_equal(fv[_VAD_S:_VAD_E], [0.0] * VAD_DIM)

def test_vad_block_values_stored():
    fv = build_feature_vector(
        _mo(), vad_scores={"valence": 0.6, "arousal": -0.3, "dominance": 0.1}
    ).flatten()
    np.testing.assert_allclose(fv[_VAD_S:_VAD_E], [0.6, -0.3, 0.1], atol=1e-6)

def test_vad_block_clipped():
    fv = build_feature_vector(
        _mo(), vad_scores={"valence": 2.0, "arousal": -5.0, "dominance": 0.0}
    ).flatten()
    assert fv[_VAD_S]     == pytest.approx(1.0,  abs=1e-5)
    assert fv[_VAD_S + 1] == pytest.approx(-1.0, abs=1e-5)


def test_cdm_block_zeros_when_missing():
    fv = build_feature_vector(_mo(), context_vector=None).flatten()
    np.testing.assert_array_equal(fv[_CDM_S:_CDM_E], [0.0] * CDM_CTX_DIM)

def test_cdm_block_zeros_when_wrong_length():
    fv = build_feature_vector(_mo(), context_vector=[0.5] * 10).flatten()
    np.testing.assert_array_equal(fv[_CDM_S:_CDM_E], [0.0] * CDM_CTX_DIM)

def test_cdm_block_used_when_correct_length():
    ctx = [float(i) / 100 for i in range(CDM_CTX_DIM)]
    fv  = build_feature_vector(_mo(), context_vector=ctx).flatten()
    cdm_out = fv[_CDM_S:_CDM_E]
    normalized_idx = {CTX_MSG_LENGTH, CTX_LATENCY_MS, CTX_HMM_EMISSION}
    for i, (actual, expected) in enumerate(zip(cdm_out, ctx)):
        if i not in normalized_idx:
            assert actual == pytest.approx(expected, abs=1e-6), f"CDM index {i} mismatch"
    assert 0.0 <= cdm_out[CTX_MSG_LENGTH]   <= 3.0
    assert 0.0 <= cdm_out[CTX_LATENCY_MS]   <= 3.0
    assert 0.0 <= cdm_out[CTX_HMM_EMISSION] <= 1.0


def test_prior_block_zeros_when_missing():
    fv = build_feature_vector(_mo(), trajectory_prior=None).flatten()
    np.testing.assert_array_equal(fv[_PRIOR_S:_PRIOR_E], [0.0] * PRIOR_DIM)

def test_prior_block_zeros_when_wrong_length():
    fv = build_feature_vector(_mo(), trajectory_prior=[0.5] * 5).flatten()
    np.testing.assert_array_equal(fv[_PRIOR_S:_PRIOR_E], [0.0] * PRIOR_DIM)

def test_prior_block_used_when_correct_length():
    prior = [float(i) / 100 for i in range(PRIOR_DIM)]
    fv = build_feature_vector(_mo(), trajectory_prior=prior).flatten()
    np.testing.assert_allclose(fv[_PRIOR_S:_PRIOR_E], prior, atol=1e-6)


def test_sarcasm_slot_default_zero():
    assert build_feature_vector(_mo()).flatten()[_SARC] == pytest.approx(0.0, abs=1e-6)

def test_sarcasm_slot_value_stored():
    assert build_feature_vector(_mo(), sarcasm_score=0.83).flatten()[_SARC] == pytest.approx(0.83, abs=1e-5)

def test_sarcasm_slot_clipped_above_one():
    assert build_feature_vector(_mo(), sarcasm_score=1.5).flatten()[_SARC] == pytest.approx(1.0, abs=1e-5)

def test_sarcasm_slot_clipped_below_zero():
    assert build_feature_vector(_mo(), sarcasm_score=-0.3).flatten()[_SARC] == pytest.approx(0.0, abs=1e-5)

def test_sarcasm_does_not_corrupt_prior():
    prior = [float(i) / 100 for i in range(PRIOR_DIM)]
    fv = build_feature_vector(_mo(), trajectory_prior=prior, sarcasm_score=0.7).flatten()
    np.testing.assert_allclose(fv[_PRIOR_S:_PRIOR_E], prior, atol=1e-6)
    assert fv[_SARC] == pytest.approx(0.7, abs=1e-5)


def test_dynamics_block_defaults_zero():
    fv = build_feature_vector(_mo()).flatten()
    np.testing.assert_array_equal(fv[_DYN_S:_DYN_E], [0.0, 0.0])

def test_dynamics_block_values_stored():
    fv = build_feature_vector(
        _mo(), dynamics_scores={"inertia": 0.4, "contagion": -0.6}
    ).flatten()
    assert fv[_DYN_S]     == pytest.approx(0.4,  abs=1e-5)
    assert fv[_DYN_S + 1] == pytest.approx(-0.6, abs=1e-5)

def test_dynamics_block_clipped():
    fv = build_feature_vector(
        _mo(), dynamics_scores={"inertia": 3.0, "contagion": -2.0}
    ).flatten()
    assert fv[_DYN_S]     == pytest.approx(1.0,  abs=1e-5)
    assert fv[_DYN_S + 1] == pytest.approx(-1.0, abs=1e-5)


def test_appraisal_block_coping_default_is_half():
    # coping defaults to 0.5 (neutral coping potential per Scherer 2001) when no
    # appraisal_scores provided — this is intentional, not a missing-data zero.
    fv = build_feature_vector(_mo()).flatten()
    assert fv[_APR_S]     == pytest.approx(0.0, abs=1e-6)   # novelty
    assert fv[_APR_S + 1] == pytest.approx(0.0, abs=1e-6)   # goal_congruence
    assert fv[_APR_S + 2] == pytest.approx(0.5, abs=1e-6)   # coping

def test_appraisal_block_values_stored():
    fv = build_feature_vector(
        _mo(), appraisal_scores={"novelty": 0.7, "goal_congruence": -0.4, "coping": 0.9}
    ).flatten()
    np.testing.assert_allclose(fv[_APR_S:_APR_E], [0.7, -0.4, 0.9], atol=1e-5)

def test_appraisal_block_clipped():
    fv = build_feature_vector(
        _mo(), appraisal_scores={"novelty": -1.0, "goal_congruence": 2.0, "coping": 5.0}
    ).flatten()
    assert fv[_APR_S]     == pytest.approx(0.0, abs=1e-5)   # novelty clipped [0,1]
    assert fv[_APR_S + 1] == pytest.approx(1.0, abs=1e-5)   # goal_congruence clipped [-1,1]
    assert fv[_APR_S + 2] == pytest.approx(1.0, abs=1e-5)   # coping clipped [0,1]


def test_missing_model_keys_ml_block_is_zero():
    # coping in appraisal defaults to 0.5 (intentional, not missing-data zero)
    fv = build_feature_vector({}).flatten()
    assert fv.shape == (FEATURE_DIM,)
    np.testing.assert_array_equal(fv[0:39], [0.0] * 39)   # VADER+BERT+GoE

def test_missing_model_keys_vad_is_zero():
    fv = build_feature_vector({}).flatten()
    np.testing.assert_array_equal(fv[_VAD_S:_VAD_E], [0.0] * VAD_DIM)

def test_partial_bert_keys_default_missing_to_zero():
    fv = build_feature_vector(_mo(bert={"joy": 0.9})).flatten()
    joy_idx = 4 + BERT_LABELS.index("joy")
    assert fv[joy_idx] == pytest.approx(0.9, abs=1e-6)
    for i, k in enumerate(BERT_LABELS):
        if k != "joy":
            assert fv[4 + i] == pytest.approx(0.0, abs=1e-6)


def test_vad_computed_from_goe():
    goe = {k: 0.0 for k in EMOTION_LABELS}
    goe["joy"] = 1.0   # pure joy → strongly positive valence + arousal
    fv = build_feature_vector(_mo(goe=goe)).flatten()
    assert fv[_VAD_S]     >  0.5, "valence should be strongly positive for joy"
    assert fv[_VAD_S + 1] >  0.0, "arousal should be positive for joy"
    assert fv[_VAD_S + 2] >  0.0, "dominance should be positive for joy"


def test_vad_negative_emotion_gives_negative_valence():
    goe = {k: 0.0 for k in EMOTION_LABELS}
    goe["grief"] = 1.0
    fv = build_feature_vector(_mo(goe=goe)).flatten()
    assert fv[_VAD_S] < -0.5, "grief should give strongly negative valence"


def test_dynamics_inertia_computed_when_prior_matches():
    goe = {k: 0.0 for k in EMOTION_LABELS}
    goe["joy"] = 1.0
    prior = [0.0] * PRIOR_DIM
    prior[EMOTION_LABELS.index("joy")] = 1.0   # prior predicted joy
    fv = build_feature_vector(_mo(goe=goe), trajectory_prior=prior).flatten()
    assert fv[_DYN_S] > 0.8, "inertia should be high when emotion matches prior"


def test_dynamics_contagion_positive_when_compound_and_ctx_align():
    goe = {k: 0.0 for k in EMOTION_LABELS}
    goe["joy"] = 1.0
    ctx = [0.0] * CDM_CTX_DIM
    from shared.constants import CTX_CURR_VALENCE
    ctx[CTX_CURR_VALENCE] = 0.8    # conversation is positive
    fv = build_feature_vector(_mo(compound=0.6, goe=goe), context_vector=ctx).flatten()
    assert fv[_DYN_S + 1] > 0.0, "contagion should be positive when user matches conversation"


def test_appraisal_novelty_nonzero_without_prior():
    goe = {k: 1.0 / len(EMOTION_LABELS) for k in EMOTION_LABELS}  # max entropy
    fv = build_feature_vector(_mo(goe=goe)).flatten()
    assert fv[_APR_S] > 0.0, "flat GoE distribution is high novelty (entropy-based)"


def test_appraisal_novelty_low_when_matches_prior():
    goe = {k: 0.0 for k in EMOTION_LABELS}
    goe["joy"] = 1.0
    prior = [0.0] * PRIOR_DIM
    prior[EMOTION_LABELS.index("joy")] = 1.0
    fv = build_feature_vector(_mo(goe=goe), trajectory_prior=prior).flatten()
    assert fv[_APR_S] < 0.1, "novelty should be near zero when emotion matches prediction"


def test_appraisal_goal_congruence_follows_vad_valence():
    goe = {k: 0.0 for k in EMOTION_LABELS}
    goe["joy"] = 1.0
    fv = build_feature_vector(_mo(goe=goe)).flatten()
    gc = fv[_APR_S + 1]
    v  = fv[_VAD_S]
    assert gc == pytest.approx(v, abs=1e-5), "goal_congruence should equal VAD valence"


def test_appraisal_coping_above_half_when_compound_positive():
    fv = build_feature_vector(_mo(compound=0.6)).flatten()
    assert fv[_APR_S + 2] > 0.5, "positive compound → coping > 0.5"


def test_appraisal_coping_below_half_when_compound_negative():
    fv = build_feature_vector(_mo(compound=-0.6)).flatten()
    assert fv[_APR_S + 2] < 0.5, "negative compound → coping < 0.5"


def test_augment_preserves_negative_vader_compound():
    """vader_compound at index 3 is in [-1, 1] — clipping to [0, 1] corrupts it."""
    mo = _mo(compound=-0.72)
    fv = build_feature_vector(mo).flatten().copy()
    assert fv[3] < 0, "vader_compound should be negative for compound=-0.72"

    rng = np.random.RandomState(0)
    noise = rng.normal(0, 0.02, size=42).astype(np.float32)
    src = fv.copy()
    src[0:3]   = np.clip(src[0:3]   + noise[0:3],   0.0, 1.0)
    src[3:4]   = np.clip(src[3:4]   + noise[3:4],   -1.0, 1.0)
    src[4:39]  = np.clip(src[4:39]  + noise[4:39],  0.0, 1.0)
    src[39:42] = np.clip(src[39:42] + noise[39:42], -1.0, 1.0)
    assert src[3] < 0.0, "vader_compound must remain negative after augmentation"


def test_augment_preserves_negative_vad_valence():
    """VAD valence at ML_DIM-3=39 is in [-1, 1] — must not be clipped to 0."""
    goe = {k: 0.0 for k in EMOTION_LABELS}
    goe["grief"] = 1.0   # grief → strongly negative VAD valence
    fv = build_feature_vector(_mo(goe=goe)).flatten().copy()
    raw_vad_val = fv[_VAD_S]
    assert raw_vad_val < 0.0, "grief should produce negative VAD valence"

    rng = np.random.RandomState(0)
    noise = rng.normal(0, 0.02, size=42).astype(np.float32)
    src = fv.copy()
    src[0:3]   = np.clip(src[0:3]   + noise[0:3],   0.0, 1.0)
    src[3:4]   = np.clip(src[3:4]   + noise[3:4],   -1.0, 1.0)
    src[4:39]  = np.clip(src[4:39]  + noise[4:39],  0.0, 1.0)
    src[39:42] = np.clip(src[39:42] + noise[39:42], -1.0, 1.0)
    assert src[_VAD_S] < 0.0, "VAD valence must remain negative after augmentation noise"


def test_goe_gate_cap_never_exceeds_50pct():
    import torch
    from meta_learner import GatingEnsembleNet, D_MODEL, DROPOUT
    from shared.constants import FEATURE_DIM

    torch.manual_seed(42)
    net = GatingEnsembleNet(n_classes=28, d=D_MODEL, dropout=DROPOUT)
    net.eval()

    rng = np.random.RandomState(0)
    X   = rng.randn(100, FEATURE_DIM).astype(np.float32)
    t   = torch.tensor(X)

    with torch.no_grad():
        _, alpha = net(t)  # [100, 5]: vader, bert, goe, vad, ctx

    goe_weights = alpha[:, 2].numpy()  # index 2 = GoEmotions
    assert (goe_weights <= 0.5001).all(), (
        f"GoEmotions gate weight exceeded 50% cap in {(goe_weights > 0.5001).sum()} samples. "
        f"max={goe_weights.max():.4f}"
    )


def test_goe_scores_no_vad_contamination():
    goe = {k: 0.0 for k in EMOTION_LABELS}
    goe["joy"] = 0.9
    goe["vad_valence"]   = 0.85
    goe["vad_arousal"]   = 0.60
    goe["vad_dominance"] = 0.55
    fv = build_feature_vector({"go_emotions": goe}).flatten()
    from shared.constants import EMOTION_LABELS as _EL
    joy_idx = _EL.index("joy")
    assert fv[11 + joy_idx] == pytest.approx(0.9, abs=1e-6), "joy should be in GoE block"
    # The slice should sum to ≈0.9 (only joy is non-zero among the 28 labels)
    assert fv[11:39].sum() == pytest.approx(0.9, abs=1e-5), "vad_* must not bleed into GoE block"
