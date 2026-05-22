"""
Tests for the meta_learner module — specifically the fallback logic and feature vector.

Run from the project root:
    python -m pytest training/test_meta_learner.py -v

These tests do NOT require the trained .pkl to exist.
They verify the fallback safety guarantees of load_meta_learner().
"""

import sys
import os
import pickle
import tempfile
import numpy as np
import pytest

# Add project root to path so shared/ and service packages are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from central_responder_service.ml.loader    import load_meta_learner
from central_responder_service.ml.predictor import build_feature_vector, predict_with_meta_learner
from shared.constants import EMOTION_LABELS, VADER_KEYS as _VADER_KEYS, BERT_LABELS as _BERT_LABELS, FEATURE_DIM


# ── load_meta_learner tests ───────────────────────────────────────────────────

class TestLoadMetaLearner:
    def test_returns_none_when_file_missing(self, tmp_path):
        """Missing file should return None, never raise."""
        result = load_meta_learner(str(tmp_path / "nonexistent.pkl"))
        assert result is None

    def test_returns_none_on_corrupt_file(self, tmp_path):
        """Corrupt/invalid pickle should return None, never raise."""
        bad_pkl = tmp_path / "bad.pkl"
        bad_pkl.write_bytes(b"this is not a pickle")
        result = load_meta_learner(str(bad_pkl))
        assert result is None

    def test_returns_none_for_non_sklearn_object(self, tmp_path):
        """A valid pickle that is not a sklearn Pipeline should return None."""
        not_a_model = {"weight": 1.0}
        bad_pkl = tmp_path / "bad_model.pkl"
        with open(bad_pkl, 'wb') as f:
            pickle.dump(not_a_model, f)
        result = load_meta_learner(str(bad_pkl))
        assert result is None

    def test_loads_valid_model(self, tmp_path):
        """A valid sklearn Pipeline should load successfully."""
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Minimal pipeline that has predict_proba
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=10))
        ])
        # Fit on tiny dummy data so predict_proba works
        X = np.random.rand(10, FEATURE_DIM)
        y = ['joy'] * 5 + ['sadness'] * 5
        pipeline.fit(X, y)

        pkl_path = tmp_path / "model.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(pipeline, f)

        result = load_meta_learner(str(pkl_path))
        assert result is not None
        assert hasattr(result, 'predict_proba')


# ── build_feature_vector tests ────────────────────────────────────────────────

class TestBuildFeatureVector:
    def test_correct_shape(self):
        """Feature vector should always be (1, FEATURE_DIM)."""
        model_outputs = {
            "vader":       {"vader_neg": 0.1, "vader_neu": 0.8, "vader_pos": 0.1, "vader_compound": 0.0},
            "basic_bert":  {"joy": 0.9, "anger": 0.05},
            "go_emotions": {"joy": 0.8, "neutral": 0.1},
            "emojinet":    {"love": 0.95},
        }
        fv = build_feature_vector(model_outputs)
        assert fv.shape == (1, FEATURE_DIM), f"Expected (1, {FEATURE_DIM}), got {fv.shape}"

    def test_missing_models_pad_zeros(self):
        """Missing model outputs should be silently padded with 0.0."""
        model_outputs = {}  # all models missing
        fv = build_feature_vector(model_outputs)
        assert fv.shape == (1, FEATURE_DIM)
        assert np.all(fv == 0.0), "All zeros expected when all models are missing"

    def test_partial_models_pad_zeros(self):
        """Only VADER present — BERT, GoEmotions, EmojiNet blocks should be zero."""
        model_outputs = {
            "vader": {"vader_neg": 0.0, "vader_neu": 1.0, "vader_pos": 0.0, "vader_compound": 0.0}
        }
        fv = build_feature_vector(model_outputs)
        assert fv.shape == (1, FEATURE_DIM)
        vader_block = fv[0, :len(_VADER_KEYS)]
        bert_block  = fv[0, len(_VADER_KEYS):len(_VADER_KEYS) + len(_BERT_LABELS)]
        assert vader_block[1] == pytest.approx(1.0)  # vader_neu
        assert np.all(bert_block == 0.0)

    def test_consistent_length_across_calls(self):
        """Same FEATURE_DIM regardless of which models have data."""
        outputs_a = {"go_emotions": {"joy": 0.9}}
        outputs_b = {"vader": {"vader_compound": -0.5}, "emojinet": {"sadness": 0.8}}
        fv_a = build_feature_vector(outputs_a)
        fv_b = build_feature_vector(outputs_b)
        assert fv_a.shape == fv_b.shape

    def test_dtype_is_float32(self):
        fv = build_feature_vector({"vader": {"vader_compound": 0.5}})
        assert fv.dtype == np.float32


# ── predict_with_meta_learner tests ──────────────────────────────────────────

class TestPredictWithMetaLearner:
    @pytest.fixture
    def tiny_model(self):
        """A minimal fitted sklearn Pipeline for testing predict logic."""
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=50))
        ])
        X = np.random.rand(20, FEATURE_DIM)
        y = ['joy'] * 10 + ['sadness'] * 10
        pipeline.fit(X, y)
        return pipeline

    def test_returns_valid_label(self, tiny_model):
        fv = np.random.rand(1, FEATURE_DIM).astype(np.float32)
        label, confidence, prob_dict, sarcasm, conflict = predict_with_meta_learner(tiny_model, fv)
        assert label in ['joy', 'sadness']
        assert 0.0 <= confidence <= 1.0
        assert isinstance(prob_dict, dict)
        assert len(prob_dict) == 2
        assert 0.0 <= sarcasm <= 1.0

    def test_returns_neutral_on_bad_input(self, tiny_model):
        """If something goes wrong (bad shape etc.), should return 'neutral', 0.0, {}."""
        bad_fv = np.array([[1, 2, 3]])  # wrong shape
        label, confidence, prob_dict, sarcasm, conflict = predict_with_meta_learner(tiny_model, bad_fv)
        assert label == "neutral"
        assert confidence == 0.0
        assert prob_dict == {}
