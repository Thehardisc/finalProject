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
from shared.utils.logger import get_logger

logger = get_logger("meta_learner")

from shared.constants import EMOTION_LABELS, VADER_KEYS, BERT_LABELS, FEATURE_DIM


# Default model path inside the container (mounted via Docker volume)
# trainer.py writes to this exact location.
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")
DEFAULT_META_PATH  = DEFAULT_MODEL_PATH.replace(".pkl", "_meta.json")


# public api

def load_meta_learner(model_path: str = DEFAULT_MODEL_PATH) -> Optional[object]:
    """
    Attempt to load the trained sklearn Pipeline from disk.

    Returns the model object if successful, or None if the file doesn't exist,
    is corrupted, or is incompatible. Never raises an exception — the caller
    should treat None as "use fallback mode".
    """
    try:
        if not os.path.exists(model_path):
            logger.warning(f"No model file at '{model_path}'. Fallback mode will be used.")
            return None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Basic sanity check — should have predict_proba (sklearn Pipeline)
        if not hasattr(model, 'predict_proba'):
            logger.warning("Loaded object is not a valid sklearn Pipeline. Fallback mode.")
            return None

        # Dimension sanity check — ensure it matches our current 103D feature vector
        try:
            dummy_x = np.zeros((1, FEATURE_DIM))
            model.predict(dummy_x)
        except Exception as e:
            logger.warning(f"Loaded model is incompatible with current {FEATURE_DIM}D features: {e}. Fallback mode.")
            return None

        # Log metadata if available
        _log_metadata()

        logger.info(f"Meta-learner loaded successfully from '{model_path}'.")
        return model

    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        logger.warning(f"Failed to load model (corrupt file?): {e}. Fallback mode.")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error loading model: {e}. Fallback mode.")
        return None


def build_feature_vector(model_outputs: dict, context: dict = None) -> np.ndarray:
    """
    Build a fixed-length float32 numpy feature vector from 4 model output dicts
    PLUS conversation context (previous mood/valence).

    Returns np.ndarray of shape (1, 103).

    Blocks:
      [0:4]    VADER (4)
      [4:11]   BERT Ekman (7)
      [11:39]  GoEmotions (28)
      [39:67]  EmojiNet (28)
      [67:96]  Context: valence(1) + one-hot prev emotion(28)
      [96:103] Derived: bert_entropy, goe_entropy, bert_margin,
                        goe_margin, bert_goe_agreement,
                        vader_abs_compound, max_goe_score
    """
    import math
    _EPS = 1e-9

    def _entropy(scores, keys):
        probs = [max(scores.get(k, 0.0), 0.0) for k in keys]
        total = sum(probs) + _EPS
        probs = [p / total for p in probs]
        return float(-sum(p * math.log(p + _EPS) for p in probs))

    def _margin(scores, keys):
        vals = sorted([scores.get(k, 0.0) for k in keys], reverse=True)
        return float(vals[0] - vals[1]) if len(vals) >= 2 else (float(vals[0]) if vals else 0.0)

    def _agreement(bert, goe):
        shared = [l for l in BERT_LABELS if l in EMOTION_LABELS]
        if not shared:
            return 0.0
        return 1.0 if max(shared, key=lambda k: bert.get(k, 0.0)) == max(shared, key=lambda k: goe.get(k, 0.0)) else 0.0

    context = context or {}
    vader_scores      = model_outputs.get("vader", {})
    bert_scores       = model_outputs.get("basic_bert", {})
    goemotions_scores = model_outputs.get("go_emotions", {})
    emoji_scores      = model_outputs.get("emojinet", {})

    vec = []

    # Block 1: VADER (4)
    for k in VADER_KEYS:
        vec.append(float(vader_scores.get(k, 0.0)))

    # Block 2: BERT Ekman (7)
    for k in BERT_LABELS:
        vec.append(float(bert_scores.get(k, 0.0)))

    # Block 3: GoEmotions (28)
    for k in EMOTION_LABELS:
        vec.append(float(goemotions_scores.get(k, 0.0)))

    # Block 4: EmojiNet (28)
    for k in EMOTION_LABELS:
        vec.append(float(emoji_scores.get(k, 0.0)))

    # Block 5: Context (29) — valence + one-hot prev emotion
    vec.append(float(context.get("avg_valence", 0.0)))
    prev_emo = context.get("prev_emotion", "neutral").lower()
    for label in EMOTION_LABELS:
        vec.append(1.0 if label == prev_emo else 0.0)

    # Block 6: Derived (7)
    vec.append(_entropy(bert_scores,       BERT_LABELS))
    vec.append(_entropy(goemotions_scores, EMOTION_LABELS))
    vec.append(_margin(bert_scores,        BERT_LABELS))
    vec.append(_margin(goemotions_scores,  EMOTION_LABELS))
    vec.append(_agreement(bert_scores, goemotions_scores))
    vec.append(abs(float(vader_scores.get("vader_compound", 0.0))))
    vec.append(max((goemotions_scores.get(k, 0.0) for k in EMOTION_LABELS), default=0.0))

    arr = np.array(vec, dtype=np.float32)
    return arr.reshape(1, -1)



def predict_with_meta_learner(
    model, feature_vector: np.ndarray
) -> Tuple[str, float, dict]:
    """
    Run inference with the meta-learner.

    Args:
        model: Loaded sklearn Pipeline (from load_meta_learner).
        feature_vector: np.ndarray of shape (1, 96) from build_feature_vector().

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
        
        # Detect conflicts (Sarcasm/Slang pivot)
        sarcasm_score, conflict_desc = detect_emotional_conflicts(feature_vector)
        
        return pred_label, confidence, all_scores, sarcasm_score, conflict_desc
    except Exception as e:
        logger.warning(f"Predict error: {e}. Returning neutral.")
        return "neutral", 0.0, {}, 0.0, None

def detect_emotional_conflicts(vec: np.ndarray):
    """
    Heuristic layer to detect 'Sentiment Flipping' (Sarcasm/Slang).
    Returns (sarcasm_score [0.0-1.0], conflict_description [str or None])
    """
    try:
        # Offsets (Must match shared/constants.py architecture)
        # VADER: 0-3 | BERT: 4-10 | GoE: 11-38 | Ctx: 39-67 | Derived: 68-74
        v_pos = vec[2]
        v_neg = vec[0]
        v_cmp = vec[3]

        bert_joy    = vec[7]   # joy in BERT Ekman (index 4=anger,5=disgust,6=fear,7=joy)
        bert_anger  = vec[4]
        bert_neutral = vec[8]  # neutral in BERT Ekman

        pos_text_signal = (v_pos + bert_joy) / 2

        sarcasm_score = 0.0
        conflict_desc = None

        # Positive text but high VADER negativity — lexical contradiction
        if pos_text_signal > 0.6 and v_neg > 0.4:
            sarcasm_score = min(pos_text_signal, v_neg) * 1.2
            conflict_desc = "Cognitive Dissonance: Positive surface text paired with high negativity signal."

        # Extreme positive compound but BERT detects anger
        elif v_cmp > 0.8 and bert_anger > 0.3:
            sarcasm_score = 0.5 + bert_anger
            conflict_desc = "Sarcasm detected: Semantic praise contradicts underlying anger signal."

        # BERT neutral with high VADER negativity — passive-aggressive pattern
        elif bert_neutral > 0.7 and v_neg > 0.25:
            sarcasm_score = 0.4
            conflict_desc = "Passive-aggression suspected: Formal neutral tone with underlying negativity."

        return min(sarcasm_score, 1.0), conflict_desc

    except Exception:
        return 0.0, None


def calculate_feature_impacts(model, feature_vector: np.ndarray, predicted_emotion: str) -> dict:
    """
    Calculate the contribution of each high-level model/context block 
    to the final prediction logic.
    """
    try:
        # 1. Access components from sklearn Pipeline
        scaler = model.named_steps['scaler']
        clf    = model.named_steps['clf']

        # 2. Scale features (to match clf's expected input)
        X_scaled = scaler.transform(feature_vector)[0]

        # 3. Resolve a linear estimator we can attribute against.
        # The trainer ships a VotingClassifier (LR + HGB + RF). Only the LR
        # sub-estimator exposes .coef_, so the impact map reflects the linear
        # component of the ensemble — the tree members can't be attributed
        # this way. Fall back to a plain estimator's .coef_ if present.
        lin = getattr(clf, 'named_estimators_', {}).get('lr', clf)
        if not hasattr(lin, 'coef_'):
            return {}

        # 4. Find index of the predicted class on the linear estimator's
        # own classes_ (matches the VotingClassifier's after a shared fit).
        classes = list(lin.classes_)
        if predicted_emotion not in classes:
            return {}
        class_idx = classes.index(predicted_emotion)

        # 5. Get coefficients for this specific class (num_features,)
        weights = lin.coef_[class_idx]
        
        # 5. Calculate raw contributions
        contributions = X_scaled * weights
        
        # 6. Group by block (matching build_feature_vector offsets)
        # Offsets: VADER(0-3), BERT(4-10), GoE(11-38), EmojiNet(39-66), Context(67-95), Derived(96-102)
        impacts = {
            "VADER":       float(np.sum(contributions[0:4])),
            "BERT":        float(np.sum(contributions[4:11])),
            "GoEmotions":  float(np.sum(contributions[11:39])),
            "EmojiNet":    float(np.sum(contributions[39:67])),
            "Context":     float(np.sum(contributions[67:96])),
            "Derived":     float(np.sum(contributions[96:103])),
        }
        
        # Normalize for visualization (Relative Importance)
        total = sum(abs(v) for v in impacts.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in impacts.items()}
        return impacts

    except Exception as e:
        logger.warning(f"Failed to calculate impacts: {e}")
        return {}


# helpers

def _log_metadata():
    """Log training metadata if the meta JSON file exists."""
    try:
        if os.path.exists(DEFAULT_META_PATH):
            with open(DEFAULT_META_PATH, 'r') as f:
                meta = json.load(f)
                logger.info(f"   Trained at      : {meta.get('trained_at', 'unknown')}")
                logger.info(f"   Training samples: {meta.get('training_samples', '?')}")
                logger.info(f"   Val accuracy    : {meta.get('validation_accuracy', '?')}")
                logger.info(f"   Test accuracy   : {meta.get('test_accuracy', '?')}")
    except Exception:
        pass  # metadata logging is best-effort
