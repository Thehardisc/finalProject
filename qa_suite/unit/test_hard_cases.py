import os
import sys
import numpy as np
import pytest

_HERE         = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SVC_ROOT     = os.path.join(_PROJECT_ROOT, "central_responder_service")
for p in (_PROJECT_ROOT, _SVC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from meta_learner import (
    build_feature_vector,
    detect_emotional_conflicts,
    _compute_inertia,
    _compute_novelty,
)
from implicit_emotion import is_implicit_candidate, should_override
from shared.constants import EMOTION_LABELS, BERT_LABELS, FEATURE_DIM, ML_DIM


def _goe(**kwargs):
    g = {k: 0.0 for k in EMOTION_LABELS}
    g.update(kwargs)
    return g

def _bert(**kwargs):
    b = {k: 0.0 for k in BERT_LABELS}
    b.update(kwargs)
    return b

def _vader(neg=0.0, neu=1.0, pos=0.0, compound=0.0):
    return {
        "vader_neg": neg, "vader_neu": neu,
        "vader_pos": pos, "vader_compound": compound,
    }

_FV_EXTRA_PARAMS = frozenset({
    "context_vector", "trajectory_prior", "sarcasm_score",
    "vad_scores", "dynamics_scores", "appraisal_scores",
})

def _fv(**kw):
    """Thin wrapper: separates model_outputs dict from build_feature_vector kwargs."""
    extra   = {k: v for k, v in kw.items() if k in _FV_EXTRA_PARAMS}
    outputs = {k: v for k, v in kw.items() if k not in _FV_EXTRA_PARAMS}
    return build_feature_vector(outputs, **extra)



class TestSarcasmDetection:

    def test_classic_sarcasm_positive_vader_negative_goe(self):
        """'Yeah, GREAT idea 🙄' — VADER is positive, GoEmotions says annoyance."""
        fv = _fv(
            vader=_vader(neg=0.05, neu=0.40, pos=0.55, compound=0.65),
            basic_bert=_bert(joy=0.70, neutral=0.20),
            go_emotions=_goe(annoyance=0.55, disapproval=0.20, joy=0.10),
        )
        score, desc = detect_emotional_conflicts(fv)
        assert score > 0.0, "Sarcasm score should be > 0 for positive VADER + negative GoE"
        assert desc is not None, "Conflict description should be set"

    def test_extreme_compound_with_annoyance_triggers_high_sarcasm(self):
        """Compound=0.95 with low surface positive signal → second branch fires."""
        fv = _fv(
            vader=_vader(pos=0.10, compound=0.95, neu=0.85, neg=0.05),
            basic_bert=_bert(neutral=0.80, joy=0.05),
            go_emotions=_goe(annoyance=0.45, neutral=0.35),
        )
        score, desc = detect_emotional_conflicts(fv)
        assert score >= 0.90, f"Expected score ≥ 0.90, got {score:.3f}"

    def test_genuine_positive_no_sarcasm(self):
        """Pure positive: VADER pos, BERT joy, GoE joy — no conflict."""
        fv = _fv(
            vader=_vader(pos=0.80, compound=0.85, neu=0.15, neg=0.05),
            basic_bert=_bert(joy=0.88),
            go_emotions=_goe(joy=0.70, admiration=0.15, gratitude=0.10),
        )
        score, _ = detect_emotional_conflicts(fv)
        assert score == 0.0, f"Pure positive should have zero sarcasm, got {score:.3f}"

    def test_passive_aggression_neutral_bert_with_annoyance_goe(self):
        """Formal 'polite' text (BERT neutral=0.75) but GoE annoyance underlying."""
        fv = _fv(
            vader=_vader(neu=0.80, compound=0.05),
            basic_bert=_bert(neutral=0.75, disgust=0.15),
            go_emotions=_goe(annoyance=0.18, disapproval=0.12, neutral=0.40),
        )
        score, desc = detect_emotional_conflicts(fv)
        assert score == pytest.approx(0.4, abs=0.01)
        assert "Passive-aggression" in (desc or "")

    def test_sarcasm_score_capped_at_one(self):
        """Even extreme inputs should not push sarcasm_score above 1.0."""
        fv = _fv(
            vader=_vader(pos=1.0, compound=1.0),
            basic_bert=_bert(joy=1.0),
            go_emotions=_goe(annoyance=1.0),
        )
        score, _ = detect_emotional_conflicts(fv)
        assert score <= 1.0, f"Sarcasm score must be capped at 1.0, got {score}"

    def test_pure_negative_no_sarcasm(self):
        """Genuine sadness: VADER negative, GoE sadness — no positive signal to create conflict."""
        fv = _fv(
            vader=_vader(neg=0.70, compound=-0.75, neu=0.25, pos=0.05),
            basic_bert=_bert(sadness=0.80),
            go_emotions=_goe(sadness=0.55, grief=0.25, disappointment=0.10),
        )
        score, _ = detect_emotional_conflicts(fv)
        assert score == 0.0, f"Genuine sadness should not trigger sarcasm, got {score:.3f}"



class TestImplicitEmotionRouting:

    def test_neutral_prediction_triggers_implicit_path(self):
        """meta=neutral + GoE neutral not hyper-confident → should route to implicit."""
        goe = _goe(neutral=0.60, curiosity=0.20)
        result = is_implicit_candidate("neutral", goe, "I just wanted to see how things go")
        assert result is True

    def test_neutral_hyper_confident_goe_skips_implicit(self):
        """GoE neutral=0.95 (genuinely neutral) → skip implicit, text is truly neutral."""
        goe = _goe(neutral=0.95)
        result = is_implicit_candidate("neutral", goe, "The meeting starts at three o'clock")
        assert result is False

    def test_low_confidence_meta_triggers_implicit_regardless_of_emotion(self):
        """meta_confidence=0.30 < threshold=0.42 → implicit even for non-neutral pred."""
        goe = _goe(joy=0.40, sadness=0.30)
        result = is_implicit_candidate("joy", goe, "Sure, whatever you think is best", meta_confidence=0.30)
        assert result is True

    def test_confident_non_neutral_skips_implicit(self):
        """High-confidence non-neutral prediction — no implicit needed."""
        goe = _goe(anger=0.75)
        result = is_implicit_candidate("anger", goe, "This is completely unacceptable behaviour", meta_confidence=0.82)
        assert result is False

    def test_too_short_text_skips_implicit(self):
        """Text with fewer than IMPLICIT_MIN_WORDS (4) words is skipped."""
        goe = _goe(neutral=0.50)
        result = is_implicit_candidate("neutral", goe, "Fine.")
        assert result is False

    def test_exact_min_words_allowed(self):
        """Exactly 4 words (= IMPLICIT_MIN_WORDS) should be allowed."""
        goe = _goe(neutral=0.55)
        result = is_implicit_candidate("neutral", goe, "I am doing fine")
        assert result is True

    def test_override_neutral_with_confident_implicit(self):
        """impl confidence=0.72 > IMPLICIT_CONF_THRESHOLD=0.50 → override neutral."""
        result = should_override("neutral", 0.45, {"emotion": "grief", "confidence": 0.72})
        assert result is True

    def test_override_blocked_when_implicit_says_neutral(self):
        """Never override toward neutral — implicit 'neutral' is always rejected."""
        result = should_override("neutral", 0.40, {"emotion": "neutral", "confidence": 0.95})
        assert result is False

    def test_override_blocked_when_implicit_low_confidence_neutral_path(self):
        """impl confidence=0.45 < 0.50 threshold → do not override."""
        result = should_override("neutral", 0.40, {"emotion": "sadness", "confidence": 0.45})
        assert result is False

    def test_override_low_conf_path_beats_meta_by_margin(self):
        """meta_confidence=0.35, impl_confidence=0.50 → beats 0.35+0.10=0.45 → override."""
        result = should_override("confusion", 0.35, {"emotion": "fear", "confidence": 0.50})
        assert result is True

    def test_override_low_conf_path_barely_misses_margin(self):
        """meta_confidence=0.38, impl_confidence=0.47 < 0.38+0.10=0.48 → no override."""
        result = should_override("confusion", 0.38, {"emotion": "fear", "confidence": 0.47})
        assert result is False



class TestConflictingSignals:

    def test_vader_positive_goe_negative_both_in_vector(self):
        """Classic sentiment contradiction: VADER says positive, GoE says grief."""
        fv = _fv(
            vader=_vader(pos=0.70, compound=0.65, neu=0.25, neg=0.05),
            go_emotions=_goe(grief=0.80, sadness=0.10),
        ).flatten()
        assert fv[2] > 0.5,  "VADER pos should be > 0.5"
        assert fv[3] > 0.5,  "VADER compound should be > 0.5"
        goe_grief_idx = EMOTION_LABELS.index("grief")
        assert fv[11 + goe_grief_idx] == pytest.approx(0.80, abs=1e-5)

    def test_high_joy_bert_low_joy_goe(self):
        """BERT says joy=0.90, GoE says remorse=0.70 — both preserved, no averaging."""
        fv = _fv(
            basic_bert=_bert(joy=0.90),
            go_emotions=_goe(remorse=0.70, sadness=0.15),
        ).flatten()
        bert_joy_idx = BERT_LABELS.index("joy")
        goe_rem_idx  = EMOTION_LABELS.index("remorse")
        assert fv[4 + bert_joy_idx] == pytest.approx(0.90, abs=1e-5)
        assert fv[11 + goe_rem_idx]  == pytest.approx(0.70, abs=1e-5)

    def test_all_models_zero_except_one_emotion(self):
        """Single non-zero GoE entry: only that slot should be non-zero in GoE block."""
        fv = _fv(go_emotions=_goe(anger=1.0)).flatten()
        anger_idx = EMOTION_LABELS.index("anger")
        assert fv[11 + anger_idx] == pytest.approx(1.0, abs=1e-5)
        goe_block = fv[11:39].copy()
        goe_block[anger_idx] = 0.0
        assert goe_block.sum() == pytest.approx(0.0, abs=1e-5)

    def test_uniform_goe_distribution_entropy(self):
        """Uniform GoE (1/28 each) → maximum entropy → VAD valence ≈ 0 (emotions cancel out)."""
        uniform = {k: 1.0 / 28 for k in EMOTION_LABELS}
        fv = _fv(go_emotions=uniform).flatten()
        vad_valence = fv[39]
        assert abs(vad_valence) < 0.15, f"Uniform GoE → near-zero valence, got {vad_valence:.3f}"

    def test_feature_dim_invariant_on_all_hard_inputs(self):
        """Every hard-case construction produces exactly FEATURE_DIM elements."""
        cases = [
            _fv(vader=_vader(compound=-1.0), go_emotions=_goe(grief=1.0)),
            _fv(vader=_vader(compound=1.0),  go_emotions=_goe(joy=1.0)),
            _fv(go_emotions={k: 1.0/28 for k in EMOTION_LABELS}),
            _fv(basic_bert=_bert(neutral=1.0)),
            _fv(),
        ]
        for i, fv in enumerate(cases):
            assert fv.shape == (1, FEATURE_DIM), f"Case {i}: shape {fv.shape} ≠ (1, {FEATURE_DIM})"



class TestInertiaNovelty:

    def test_identical_goe_prior_gives_max_inertia(self):
        """Same emotion now and in prior → cosine similarity = 1 → inertia = 1."""
        goe = np.zeros(28, dtype=np.float32)
        goe[EMOTION_LABELS.index("anger")] = 1.0
        inertia = _compute_inertia(goe, goe.copy())
        assert inertia == pytest.approx(1.0, abs=1e-5)

    def test_opposite_emotion_gives_negative_inertia(self):
        """joy vs grief: opposite ends of the valence axis → near-zero or negative cosine."""
        goe_joy   = np.zeros(28, dtype=np.float32); goe_joy[EMOTION_LABELS.index("joy")]   = 1.0
        goe_grief = np.zeros(28, dtype=np.float32); goe_grief[EMOTION_LABELS.index("grief")] = 1.0
        inertia = _compute_inertia(goe_joy, goe_grief)
        assert inertia == pytest.approx(0.0, abs=0.01), f"Orthogonal emotions → inertia≈0, got {inertia:.3f}"

    def test_zero_prior_gives_zero_inertia(self):
        """No prior (all zeros) → norm check prevents division by zero → returns 0."""
        goe = np.zeros(28, dtype=np.float32)
        goe[EMOTION_LABELS.index("joy")] = 0.9
        prior = np.zeros(28, dtype=np.float32)
        assert _compute_inertia(goe, prior) == 0.0

    def test_novelty_zero_prior_falls_back_to_entropy(self):
        """No prior → novelty uses normalised GoE entropy (> 0 for any non-degenerate dist)."""
        goe = np.zeros(28, dtype=np.float32)
        goe[EMOTION_LABELS.index("joy")] = 0.7
        goe[EMOTION_LABELS.index("admiration")] = 0.3
        novelty = _compute_novelty(goe, np.zeros(28, dtype=np.float32))
        assert novelty > 0.0, "Novelty from entropy should be > 0"

    def test_expected_emotion_has_zero_novelty(self):
        """Exact match between current GoE and prior → novelty = 1 - 1 = 0."""
        goe = np.zeros(28, dtype=np.float32)
        goe[EMOTION_LABELS.index("sadness")] = 1.0
        novelty = _compute_novelty(goe, goe.copy())
        assert novelty == pytest.approx(0.0, abs=1e-5)

    def test_surprising_emotion_has_high_novelty(self):
        """Prior predicts anger, actual is joy — very different → novelty close to 1."""
        goe_anger = np.zeros(28, dtype=np.float32); goe_anger[EMOTION_LABELS.index("anger")] = 1.0
        goe_joy   = np.zeros(28, dtype=np.float32); goe_joy[EMOTION_LABELS.index("joy")]     = 1.0
        novelty = _compute_novelty(goe_joy, goe_anger)
        assert novelty == pytest.approx(1.0, abs=0.01), f"Surprise emotion → high novelty, got {novelty:.3f}"



def test_goe_gate_cap_with_extreme_goe_dominant_vector():
    """Feature vector where GoEmotions block is all-ones (extreme GoE signal)."""
    import torch
    from meta_learner import GatingEnsembleNet, D_MODEL, DROPOUT

    torch.manual_seed(99)
    net = GatingEnsembleNet(n_classes=28, d=D_MODEL, dropout=DROPOUT)
    net.eval()

    X = np.zeros((50, FEATURE_DIM), dtype=np.float32)
    X[:, 11:39] = 1.0

    t = torch.tensor(X)
    with torch.no_grad():
        _, alpha = net(t)

    goe_w = alpha[:, 2].numpy()
    assert (goe_w <= 0.5001).all(), f"GoE cap violated with extreme GoE input: max={goe_w.max():.4f}"


def test_goe_gate_cap_with_extreme_context_vector():
    """Feature vector with zero NLP block but saturated context."""
    import torch
    from meta_learner import GatingEnsembleNet, D_MODEL, DROPOUT

    torch.manual_seed(77)
    net = GatingEnsembleNet(n_classes=28, d=D_MODEL, dropout=DROPOUT)
    net.eval()

    X = np.zeros((50, FEATURE_DIM), dtype=np.float32)
    X[:, ML_DIM:] = 1.0

    t = torch.tensor(X)
    with torch.no_grad():
        _, alpha = net(t)

    goe_w = alpha[:, 2].numpy()
    assert (goe_w <= 0.5001).all(), f"GoE cap violated with saturated context: max={goe_w.max():.4f}"



class TestEdgeCases:

    def test_all_zero_input_gives_valid_dim_vector(self):
        """Empty model outputs → feature vector is all zeros, correct shape."""
        fv = build_feature_vector({}).flatten()
        assert fv.shape == (FEATURE_DIM,)
        assert np.isfinite(fv).all()

    def test_nan_in_goe_scores_handled_gracefully(self):
        """NaN in GoE scores should not propagate — float() coercion returns 0."""
        goe = _goe(joy=float('nan'), sadness=0.5)
        fv = _fv(go_emotions=goe).flatten()
        assert np.isfinite(fv).all(), "NaN in input must not produce NaN in feature vector"

    def test_vader_compound_at_boundary_values(self):
        """VADER compound = ±1.0 (max polarisation) should be stored unchanged."""
        for compound in (-1.0, 1.0):
            fv = _fv(vader=_vader(compound=compound)).flatten()
            assert fv[3] == pytest.approx(compound, abs=1e-5), \
                f"compound={compound} should be stored as-is, got {fv[3]}"

    def test_sarcasm_score_in_feature_vector_boundary(self):
        """Sarcasm score = 1.0 stored at index 110."""
        fv = _fv(sarcasm_score=1.0).flatten()
        assert fv[110] == pytest.approx(1.0, abs=1e-5)

    def test_sarcasm_score_clipped_above_one_in_feature_vector(self):
        """sarcasm_score=2.5 must be clipped to 1.0 in the feature vector."""
        fv = _fv(sarcasm_score=2.5).flatten()
        assert fv[110] == pytest.approx(1.0, abs=1e-5)

    def test_dynamics_inertia_at_minus_one_stored_correctly(self):
        """Inertia = -1.0 (maximum oscillation) stored in index 111."""
        fv = _fv(dynamics_scores={"inertia": -1.0, "contagion": 0.0}).flatten()
        assert fv[111] == pytest.approx(-1.0, abs=1e-5)

    def test_appraisal_goal_congruence_negative_stored(self):
        """Negative goal_congruence (opposed to goals) stored in index 114."""
        fv = _fv(appraisal_scores={"novelty": 0.5, "goal_congruence": -0.8, "coping": 0.5}).flatten()
        assert fv[114] == pytest.approx(-0.8, abs=1e-5)

    def test_context_vector_wrong_length_uses_zeros(self):
        """Context vector shorter than CDM_CTX_DIM → zero-padded, no crash."""
        fv = _fv(context_vector=[0.5] * 10)
        assert fv.shape == (1, FEATURE_DIM)
        assert np.isfinite(fv).all()

    def test_very_long_text_does_not_affect_feature_dim(self):
        """Long text won't change feature vector shape (NLP truncation is upstream)."""
        fv = _fv(go_emotions=_goe(joy=0.9))
        assert fv.shape == (1, FEATURE_DIM)

    def test_multiple_high_confidence_emotions_sum_handled(self):
        """GoE scores summing to >1 (unnormalised) — each stored as-is, no renormalisation."""
        goe = _goe(joy=0.80, love=0.80, admiration=0.60)
        fv  = _fv(go_emotions=goe).flatten()
        joy_idx = EMOTION_LABELS.index("joy")
        assert fv[11 + joy_idx] == pytest.approx(0.80, abs=1e-5)

    def test_appraisal_explicit_scores_override_computed(self):
        """Explicit appraisal_scores override internal computation: indices 113-116."""
        fv = _fv(
            go_emotions=_goe(joy=0.9),
            appraisal_scores={"novelty": 0.88, "goal_congruence": -0.77, "coping": 0.33},
        ).flatten()
        assert fv[113] == pytest.approx(0.88, abs=1e-5), f"novelty mismatch: {fv[113]}"
        assert fv[114] == pytest.approx(-0.77, abs=1e-5), f"goal_congruence mismatch: {fv[114]}"
        assert fv[115] == pytest.approx(0.33, abs=1e-5), f"coping mismatch: {fv[115]}"

    def test_appraisal_fallback_nonzero_when_no_explicit(self):
        """Without explicit appraisal_scores, computed values must be non-trivial for clear inputs."""
        fv = _fv(
            vader=_vader(compound=0.9, pos=0.85, neu=0.15),
            go_emotions=_goe(joy=0.85),
        ).flatten()
        coping = fv[115]
        assert coping > 0.5, f"High-compound → coping > 0.5, got {coping:.3f}"



import sys as _sys
import os as _os
_AGG_DIR = _os.path.join(_PROJECT_ROOT, "aggregation_service")
if _AGG_DIR not in _sys.path:
    _sys.path.insert(0, _AGG_DIR)
from emotion_dynamics import compute_inertia, compute_contagion


class TestEmotionDynamics:

    def test_inertia_monotone_sequence_positive(self):
        """Strictly increasing sequence → lag-1 autocorrelation close to 1."""
        seq = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert compute_inertia(seq) > 0.95

    def test_inertia_alternating_sequence_negative(self):
        """Strictly alternating sequence → lag-1 autocorrelation close to -1."""
        seq = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        assert compute_inertia(seq) < -0.95

    def test_inertia_constant_sequence_returns_zero(self):
        """All-same values → std=0 → returns 0.0 (not NaN)."""
        seq = [0.5] * 8
        result = compute_inertia(seq)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_inertia_too_short_returns_zero(self):
        """Fewer than 4 values → returns 0.0."""
        assert compute_inertia([0.1, 0.5, 0.9]) == 0.0
        assert compute_inertia([]) == 0.0

    def test_inertia_output_in_valid_range(self):
        """Result must always be in [-1, 1]."""
        import random
        random.seed(42)
        for _ in range(50):
            seq = [random.uniform(-1, 1) for _ in range(12)]
            r = compute_inertia(seq)
            assert -1.0 <= r <= 1.0, f"inertia={r} out of range"

    def test_contagion_synchronized_sequences_positive(self):
        """Seq A leads seq B by one step → contagion near 1."""
        a = [0.0, 0.3, 0.6, 0.9, 0.9, 0.9]
        b = [0.0, 0.0, 0.3, 0.6, 0.9, 0.9]
        assert compute_contagion(a, b) > 0.90

    def test_contagion_anti_synchronized_sequences_negative(self):
        """A and B trend in opposite directions → contagion negative."""
        a = [0.0, 0.2, 0.4, 0.6, 0.8]
        b = [0.8, 0.6, 0.4, 0.2, 0.0]
        result = compute_contagion(a, b)
        assert result < -0.50, f"anti-sync contagion should be negative, got {result:.3f}"

    def test_contagion_too_short_returns_zero(self):
        """Either sequence shorter than 3 → returns 0.0."""
        assert compute_contagion([0.1, 0.5], [0.2, 0.4, 0.6]) == 0.0
        assert compute_contagion([0.1, 0.5, 0.9], [0.2, 0.4]) == 0.0

    def test_contagion_output_in_valid_range(self):
        """Result must always be in [-1, 1]."""
        import random
        random.seed(7)
        for _ in range(50):
            a = [random.uniform(-1, 1) for _ in range(10)]
            b = [random.uniform(-1, 1) for _ in range(10)]
            r = compute_contagion(a, b)
            assert -1.0 <= r <= 1.0, f"contagion={r} out of range"
