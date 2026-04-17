"""
meta_learner.py — Meta-Learner module for the Central Responder Service.

Provides:
  - load_meta_learner()          Load trained .pkl, return None on any failure
  - build_feature_vector()       Assemble fixed-length input from 4 model dicts
  - predict_with_meta_learner()  Run inference, return (dominant_emotion, confidence)

The Central Responder calls these functions. If load_meta_learner() returns None,
the service exits — a trained model is strictly required.
"""

import os
import pickle
import json
import numpy as np
from typing import Optional, Tuple

# ── Label & feature constants (must mirror train_meta_learner.py) ─────────────
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Fixed ordering for VADER and BERT blocks — must match training script
_VADER_KEYS  = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
_BERT_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Total feature vector length: VADER(4) + BERT(7) + GoEmotions(28) + EmojiNet(28) = 67
FEATURE_DIM = len(_VADER_KEYS) + len(_BERT_LABELS) + len(EMOTION_LABELS) + len(EMOTION_LABELS)


# Default model path inside the container (mounted via Docker volume)
# trainer.py writes to this exact location.
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")
DEFAULT_META_PATH  = DEFAULT_MODEL_PATH.replace(".pkl", "_meta.json")


# ── Public API ────────────────────────────────────────────────────────────────

def load_meta_learner(model_path: str = DEFAULT_MODEL_PATH) -> Optional[object]:
    """
    Attempt to load the trained sklearn Pipeline from disk.

    Returns the model object if successful, or None if the file doesn't exist,
    is corrupted, or is incompatible. Never raises an exception — the caller
    should treat None as "use fallback mode".
    """
    try:
        if not os.path.exists(model_path):
            print(f"[MetaLearner] [WARN] No model file at '{model_path}'. Fallback mode will be used.")
            return None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Basic sanity check — should have predict_proba (sklearn Pipeline)
        if not hasattr(model, 'predict_proba'):
            print("[MetaLearner] [WARN] Loaded object is not a valid sklearn Pipeline. Fallback mode.")
            return None

        # Log metadata if available
        _log_metadata()

        print(f"[MetaLearner] [OK] Meta-learner loaded successfully from '{model_path}'.")
        return model

    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        print(f"[MetaLearner] [WARN] Failed to load model (corrupt file?): {e}. Fallback mode.")
        return None
    except Exception as e:
        print(f"[MetaLearner] [WARN] Unexpected error loading model: {e}. Fallback mode.")
        return None


def build_feature_vector(model_outputs: dict) -> np.ndarray:
    """
    Build a fixed-length float32 numpy feature vector from the 4 model output dicts.

    Args:
        model_outputs: dict keyed by model name, e.g.:
            {
                "vader":       {"vader_neg": 0.1, "vader_neu": 0.7, ...},
                "basic_bert":  {"anger": 0.05, "joy": 0.8, ...},
                "go_emotions": {"joy": 0.7, "neutral": 0.1, ...},
                "emojinet":    {"love": 0.95, ...}   # may be empty {}
            }

    Returns:
        np.ndarray of shape (67,) — consistent length regardless of which
        models have partial or missing outputs.
    """
    vader_scores      = model_outputs.get("vader", {})
    bert_scores       = model_outputs.get("basic_bert", {})
    goemotions_scores = model_outputs.get("go_emotions", {})
    emojinet_scores   = model_outputs.get("emojinet", {})

    vec = []

    # Block 1: VADER (4 dims)
    for k in _VADER_KEYS:
        vec.append(float(vader_scores.get(k, 0.0)))

    # Block 2: BERT Ekman (7 dims)
    for k in _BERT_LABELS:
        vec.append(float(bert_scores.get(k, 0.0)))

    # Block 3: GoEmotions (28 dims, fixed order)
    for k in EMOTION_LABELS:
        vec.append(float(goemotions_scores.get(k, 0.0)))

    # Block 4: EmojiNet mapped to GoEmotions space (28 dims, zeros if no emoji)
    for k in EMOTION_LABELS:
        vec.append(float(emojinet_scores.get(k, 0.0)))

    return np.array(vec, dtype=np.float32).reshape(1, -1)  # shape (1, 67)


def predict_with_meta_learner(
    model, feature_vector: np.ndarray
) -> Tuple[str, float, dict]:
    """
    Run inference with the meta-learner.

    Args:
        model: Loaded sklearn Pipeline (from load_meta_learner).
        feature_vector: np.ndarray of shape (1, 67) from build_feature_vector().

    Returns:
        (dominant_emotion: str, confidence: float, all_scores: dict)
        On any error, returns ("neutral", 0.0, {}) — never raises.
    """
    try:
        pred_label = model.predict(feature_vector)[0]
        proba = model.predict_proba(feature_vector)[0]
        classes = model.classes_
        
        # Build the full distribution dict
        all_scores = {str(k): float(v) for k, v in zip(classes, proba)}
        
        label_idx = list(classes).index(pred_label)
        confidence = float(proba[label_idx])
        return pred_label, confidence, all_scores
    except Exception as e:
        print(f"[MetaLearner] [WARN] Predict error: {e}. Returning neutral.")
        return "neutral", 0.0, {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_metadata():
    """Log training metadata if the meta JSON file exists."""
    try:
        if os.path.exists(DEFAULT_META_PATH):
            with open(DEFAULT_META_PATH, 'r') as f:
                meta = json.load(f)
            print(f"[MetaLearner]    Trained at      : {meta.get('trained_at', 'unknown')}")
            print(f"[MetaLearner]    Training samples: {meta.get('training_samples', '?')}")
            print(f"[MetaLearner]    Val accuracy    : {meta.get('validation_accuracy', '?')}")
            print(f"[MetaLearner]    Test accuracy   : {meta.get('test_accuracy', '?')}")
    except Exception:
        pass  # metadata logging is best-effort
