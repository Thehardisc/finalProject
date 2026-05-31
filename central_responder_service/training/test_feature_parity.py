"""
Parity test: the live inference path (ml/predictor.py:build_feature_vector) and
the training path (trainer/preprocessor.py:build_fv) MUST produce identical 103-dim
feature vectors for the same input.

CLAUDE.md flags this as the #1 invariant of the system: a mismatch causes
`X has N features but StandardScaler expecting M` errors at runtime.

This test catches any future drift between the two implementations.

Run from project root:
    python -m pytest central_responder_service/training/test_feature_parity.py -v
"""
import os
import sys
import importlib.util
import numpy as np
import pytest

# Add project root + central_responder_service to sys.path so the production
# imports (which use `from ml.features import ...`, etc.) resolve.
_HERE         = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SVC_ROOT     = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_PROJECT_ROOT, _SVC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.predictor       import build_feature_vector as live_build_fv
from shared.constants   import EMOTION_LABELS, BERT_LABELS, VADER_KEYS, FEATURE_DIM


def _load_preprocessor():
    """
    Load trainer/preprocessor.py directly without executing trainer/__init__.py.

    The package __init__ imports runner.py which pulls heavy runtime deps
    (redis, sklearn analyzers) that the parity test doesn't need.
    """
    path = os.path.join(_SVC_ROOT, "trainer", "preprocessor.py")
    spec = importlib.util.spec_from_file_location("trainer_preprocessor_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_build_fv = _load_preprocessor().build_fv


# ── Test cases ────────────────────────────────────────────────────────────────

def _vader_dict(neg=0.0, neu=0.0, pos=0.0, compound=0.0):
    """Both implementations key VADER differently. Provide both naming styles."""
    return {
        "vader_neg":      neg,
        "vader_neu":      neu,
        "vader_pos":      pos,
        "vader_compound": compound,
        # Inference build expects raw keys; trainer prefixes them — provide both
        "neg":      neg,
        "neu":      neu,
        "pos":      pos,
        "compound": compound,
    }


def _empty_emotion_dict():
    return {label: 0.0 for label in EMOTION_LABELS}


def _empty_bert_dict():
    return {label: 0.0 for label in BERT_LABELS}


CASES = [
    pytest.param(
        {
            "vader":       _vader_dict(),
            "basic_bert":  _empty_bert_dict(),
            "go_emotions": _empty_emotion_dict(),
            "emojinet":    _empty_emotion_dict(),
        },
        {"avg_valence": 0.0, "prev_emotion": "neutral"},
        id="all-zero",
    ),
    pytest.param(
        {
            "vader":       _vader_dict(neg=0.1, neu=0.7, pos=0.2, compound=0.4),
            "basic_bert":  {**_empty_bert_dict(), "joy": 0.85, "neutral": 0.10},
            "go_emotions": {**_empty_emotion_dict(), "joy": 0.78, "amusement": 0.15},
            "emojinet":    {**_empty_emotion_dict(), "joy": 0.6, "love": 0.3},
        },
        {"avg_valence": 0.5, "prev_emotion": "joy"},
        id="strong-joy",
    ),
    pytest.param(
        {
            "vader":       _vader_dict(neg=0.6, neu=0.3, pos=0.1, compound=-0.7),
            "basic_bert":  {**_empty_bert_dict(), "anger": 0.7, "sadness": 0.2},
            "go_emotions": {**_empty_emotion_dict(), "anger": 0.6, "annoyance": 0.25, "disappointment": 0.1},
            "emojinet":    _empty_emotion_dict(),
        },
        {"avg_valence": -0.3, "prev_emotion": "sadness"},
        id="negative-no-emoji",
    ),
    pytest.param(
        {
            "vader":       _vader_dict(neg=0.33, neu=0.34, pos=0.33, compound=0.0),
            "basic_bert":  {label: 1.0 / len(BERT_LABELS)    for label in BERT_LABELS},
            "go_emotions": {label: 1.0 / len(EMOTION_LABELS) for label in EMOTION_LABELS},
            "emojinet":    {label: 1.0 / len(EMOTION_LABELS) for label in EMOTION_LABELS},
        },
        {"avg_valence": 0.0, "prev_emotion": "neutral"},
        id="uniform",
    ),
    pytest.param(
        {
            "vader":       _vader_dict(neg=0.0, neu=0.5, pos=0.5, compound=0.6),
            "basic_bert":  {**_empty_bert_dict(), "joy": 0.5},
            "go_emotions": {**_empty_emotion_dict(), "love": 0.7},
            "emojinet":    {**_empty_emotion_dict(), "love": 0.9, "joy": 0.6, "admiration": 0.4},
        },
        {"avg_valence": 0.8, "prev_emotion": "love"},
        id="emoji-heavy",
    ),
    pytest.param(
        {
            "vader":       _vader_dict(neg=0.05, neu=0.9, pos=0.05, compound=0.1),
            "basic_bert":  {**_empty_bert_dict(), "neutral": 0.95},
            "go_emotions": {**_empty_emotion_dict(), "neutral": 0.9, "curiosity": 0.08},
            "emojinet":    _empty_emotion_dict(),
        },
        {},  # missing context — both impls should default to avg_valence=0.0, prev='neutral'
        id="missing-context",
    ),
]


# ── The parity assertion ──────────────────────────────────────────────────────


def _trainer_inputs(model_outputs, context):
    """
    The trainer's build_fv takes the four dicts directly (vader, bert, goe, emoji)
    rather than the model_outputs nested structure. Convert.

    Trainer uses VADER keys with the "vader_" prefix; live uses unprefixed too.
    The _vader_dict helper above provides both so both impls see the same numbers.
    """
    return (
        model_outputs["vader"],
        model_outputs["basic_bert"],
        model_outputs["go_emotions"],
        model_outputs["emojinet"],
        context,
    )


@pytest.mark.parametrize("model_outputs,context", CASES)
def test_feature_vector_parity(model_outputs, context):
    """Live build_feature_vector and trainer build_fv must agree for every case."""
    inference_fv = live_build_fv(model_outputs, context=context).flatten()

    vader, bert, goe, emoji, ctx = _trainer_inputs(model_outputs, context)
    training_fv = train_build_fv(vader, bert, goe, emoji, context=ctx)

    assert inference_fv.shape == (FEATURE_DIM,)
    assert training_fv.shape  == (FEATURE_DIM,)

    assert np.allclose(inference_fv, training_fv, atol=1e-7), (
        f"FEATURE PARITY VIOLATION — first 5 deltas at indices "
        f"{np.argsort(-np.abs(inference_fv - training_fv))[:5].tolist()}\n"
        f"  inference: {inference_fv[:8]}\n"
        f"  training : {training_fv[:8]}"
    )


def test_block_offsets_match_documented_layout():
    """Sanity-check that the layout matches CLAUDE.md: 4+7+28+28+29+7 = 103."""
    assert FEATURE_DIM == 103
    assert len(VADER_KEYS)    == 4
    assert len(BERT_LABELS)   == 7
    assert len(EMOTION_LABELS) == 28
