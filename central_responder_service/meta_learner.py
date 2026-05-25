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

        # Dimension sanity check — ensure it matches our current 190D feature vector
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


def build_feature_vector(model_outputs: dict, context_vector: list = None) -> np.ndarray:
    """
    Build a fixed-length float32 numpy feature vector.
    Returns np.ndarray of shape (1, 190).

    Layout:
      [0:4]    VADER (4)        ─── ML block (39 dims)
      [4:11]   BERT Ekman (7)   │
      [11:39]  GoEmotions (28)  ─
      [39:190] Context Engine (151) — CDM states, trajectory, embedding[:128]
    """
    vader_scores      = model_outputs.get("vader", {})
    bert_scores       = model_outputs.get("basic_bert", {})
    goemotions_scores = model_outputs.get("go_emotions", {})

    vec = []

    # VADER (4)
    for k in VADER_KEYS:
        vec.append(float(vader_scores.get(k, 0.0)))

    # BERT Ekman (7)
    for k in BERT_LABELS:
        vec.append(float(bert_scores.get(k, 0.0)))

    # GoEmotions (28)
    for k in EMOTION_LABELS:
        vec.append(float(goemotions_scores.get(k, 0.0)))

    # Context Engine (151) — CDM state machine + trajectory + embedding[:128]
    ctx = context_vector if (context_vector and len(context_vector) == 151) else [0.0] * 151
    vec.extend(ctx)

    arr = np.array(vec, dtype=np.float32)
    return arr.reshape(1, -1)



def predict_with_meta_learner(
    model, feature_vector: np.ndarray
) -> Tuple[str, float, dict]:
    """
    Run inference with the meta-learner.

    Args:
        model: Loaded sklearn Pipeline (from load_meta_learner).
        feature_vector: np.ndarray of shape (1, 190) from build_feature_vector().

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
        # Flatten (1, FEATURE_DIM) → (FEATURE_DIM,) so scalar indexing works
        v = vec.flatten()
        # Offsets: VADER[0:4] BERT[4:11] GoE[11:39] Context[39:190]
        v_pos = float(v[2])    # VADER pos
        v_neg = float(v[0])    # VADER neg
        v_cmp = float(v[3])    # VADER compound

        bert_joy   = float(v[7])   # BERT joy   (BERT_LABELS index 3)
        bert_anger = float(v[4])   # BERT anger (BERT_LABELS index 0)

        # GoEmotions block [11:39] — EMOTION_LABELS: annoyance=3, disapproval=10, disgust=11
        emo_annoyance   = float(v[11 + 3])
        emo_disapproval = float(v[11 + 10])
        emo_disgust     = float(v[11 + 11])
        
        neg_emo_signal = max(emo_annoyance, emo_disapproval, emo_disgust)
        pos_text_signal = (v_pos + bert_joy) / 2
        
        sarcasm_score = 0.0
        conflict_desc = None
        
        # Signature: Positive Text + Eye-roll/Negative Emoji
        if pos_text_signal > 0.6 and neg_emo_signal > 0.4:
            sarcasm_score = min(pos_text_signal, neg_emo_signal) * 1.2 # Boost score
            conflict_desc = "Cognitive Dissonance: High-fidelity positive text paired with dismissive visual cues."
        
        # Signature: Extreme Positive + Flip
        elif v_cmp > 0.8 and neg_emo_signal > 0.2:
             sarcasm_score = 0.5 + neg_emo_signal
             conflict_desc = "Sarcasm detected: Semantic praise contradicts visual frustration."
             
        # Signature: Passive Aggressive (Neutral BERT + Low intensity neg emoji)
        elif v[8] > 0.7 and (emo_annoyance > 0.1 or emo_disapproval > 0.1):
             sarcasm_score = 0.4
             conflict_desc = "Passive-aggression suspected: Formal 'Neutral' text with underlying emoji tension."

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
        
        # 3. Find index of the predicted class
        classes = list(clf.classes_)
        if predicted_emotion not in classes:
            return {}
        class_idx = classes.index(predicted_emotion)
        
        # 4. Get coefficients for this specific class (num_features,)
        # For binary case clf.coef_ is (1, d), for multi (c, d)
        weights = clf.coef_[class_idx]
        
        # 5. Calculate raw contributions
        contributions = X_scaled * weights
        
        # 6. Group by block — offsets match FEATURE_DIM=190
        # VADER[0:4] BERT[4:11] GoE[11:39] Context[39:190]
        impacts = {
            "VADER":      float(np.sum(contributions[0:4])),
            "BERT":       float(np.sum(contributions[4:11])),
            "GoEmotions": float(np.sum(contributions[11:39])),
            "Context":    float(np.sum(contributions[39:190])),
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
