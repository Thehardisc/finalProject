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

from shared.constants import EMOTION_LABELS, VADER_KEYS, BERT_LABELS, FEATURE_DIM, CONTEXT_DIM, ML_DIM

# ── Context correction constants ───────────────────────────────────────────────

_MAX_CTX_WEIGHT     = float(os.environ.get("CONTEXT_WEIGHT", "0.45"))
_MISMATCH_CTX_CAP   = float(os.environ.get("CONTEXT_WEIGHT_MISMATCH", "0.65"))

_POSITIVE_EMOTIONS = frozenset({
    'admiration', 'amusement', 'approval', 'caring', 'curiosity',
    'desire', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief',
})
_NEGATIVE_EMOTIONS = frozenset({
    'anger', 'annoyance', 'disapproval', 'disgust', 'disappointment',
    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness',
})

# Previous-emotion → valence used to amplify polarity when cur_valence is weak
_PREV_EMOTION_VALENCE: dict = {
    'anger': -0.8, 'annoyance': -0.6, 'disapproval': -0.7, 'disgust': -0.8,
    'disappointment': -0.6, 'embarrassment': -0.5, 'fear': -0.7, 'grief': -0.9,
    'nervousness': -0.5, 'remorse': -0.7, 'sadness': -0.7,
    'admiration': 0.8,  'amusement': 0.6,  'approval': 0.6,  'caring': 0.7,
    'curiosity': 0.3,   'desire': 0.5,     'excitement': 0.8, 'gratitude': 0.8,
    'joy': 0.9,         'love': 0.9,       'optimism': 0.7,   'pride': 0.7,
    'relief': 0.6,      'neutral': 0.0,    'confusion': -0.1, 'realization': 0.1,
    'surprise': 0.2,
}


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

        # Dimension sanity check — ensure it matches our current 62D feature vector
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
    Returns np.ndarray of shape (1, 62).

    Layout:
      [0:4]   VADER (4)        ─── ML block (39 dims)
      [4:11]  BERT Ekman (7)   │
      [11:39] GoEmotions (28)  ─
      [39:62] Context Engine (23) — CDM scalars only
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

    # Context Engine (23) — CDM scalars only
    ctx = context_vector if (context_vector and len(context_vector) == CONTEXT_DIM) else [0.0] * CONTEXT_DIM
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
        feature_vector: np.ndarray of shape (1, 62) from build_feature_vector().

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
        # Offsets: VADER[0:4] BERT[4:11] GoE[11:39] Context[39:62]
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
    Block-level feature attribution via leave-one-out sensitivity.

    Measures how much the predicted-class probability drops when each block
    is zeroed out. Model-agnostic — works for MLP, LR, or any sklearn Pipeline.
    4 forward passes total; negligible inference latency.
    """
    try:
        classes = list(model.classes_)
        if predicted_emotion not in classes:
            return {}
        class_idx = classes.index(predicted_emotion)

        base_proba = float(model.predict_proba(feature_vector)[0][class_idx])

        # Block offsets: VADER[0:4] BERT[4:11] GoE[11:39] Context[39:FEATURE_DIM]
        blocks = {
            "VADER":      slice(0,     4),
            "BERT":       slice(4,     11),
            "GoEmotions": slice(11,    ML_DIM),
            "Context":    slice(ML_DIM, FEATURE_DIM),
        }

        impacts = {}
        for name, sl in blocks.items():
            x_masked = feature_vector.copy()
            x_masked[0, sl] = 0.0
            masked_proba = float(model.predict_proba(x_masked)[0][class_idx])
            impacts[name] = abs(base_proba - masked_proba)

        total = sum(impacts.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in impacts.items()}
        return {k: 0.25 for k in impacts}

    except Exception as e:
        logger.warning(f"Failed to calculate impacts: {e}")
        return {}


def apply_context_correction(
    scores: dict,
    context_vector: list,
    prev_emotion: str = "neutral",
) -> Tuple[dict, float]:
    """
    Post-prediction context correction via additive blending.

    Polarity is derived from EMA valence (cv[20]), velocity (cv[15]),
    historical valence (cv[17]), and the previous-turn dominant emotion.
    CDM state is intentionally excluded — it tracks conversation trajectory,
    not valence, and can contradict EMA valence (e.g. velocity-driven
    ASCENDING_POSITIVE in a deeply negative conversation).

    When polarity opposes the dominant emotion (mismatch), blending is
    amplified up to CONTEXT_WEIGHT_MISMATCH (default 0.65).

    Returns (corrected_scores, effective_weight).
    effective_weight == 0.0 means signal was too weak — scores unchanged.
    """
    if not context_vector or len(context_vector) < 21:
        return scores, 0.0

    cv = context_vector
    cur_valence  = float(np.clip(cv[20], -1.0, 1.0))  # EMA VADER compound
    velocity     = float(np.clip(cv[15], -1.0, 1.0))  # Δvalence
    hist_valence = float(np.clip(cv[17], -1.0, 1.0))  # Qdrant episodic baseline
    prev_val     = _PREV_EMOTION_VALENCE.get(prev_emotion, 0.0)

    # Valence-only polarity — no CDM state
    ctx_polarity = (
        cur_valence  * 0.40 +
        velocity     * 0.15 +
        hist_valence * 0.15 +
        prev_val     * 0.30
    )

    ctx_strength = abs(ctx_polarity)
    if ctx_strength < 0.08:
        return scores, 0.0

    # Target distribution: all mass on the correct-polarity emotion pool,
    # weighted by existing ML scores within that pool.
    if ctx_polarity < 0:
        pool = {e: float(v) for e, v in scores.items() if e in _NEGATIVE_EMOTIONS}
    else:
        pool = {e: float(v) for e, v in scores.items() if e in _POSITIVE_EMOTIONS}

    pool_total = sum(pool.values())
    if pool_total < 1e-9:
        # No existing mass in target pool — uniform fallback
        target_set = _NEGATIVE_EMOTIONS if ctx_polarity < 0 else _POSITIVE_EMOTIONS
        pool = {e: 1.0 for e in target_set if e in scores}
        pool_total = float(len(pool)) or 1.0

    target = {e: 0.0 for e in scores}
    for e, v in pool.items():
        target[e] = v / pool_total

    # Mismatch: dominant opposes context polarity → amplify correction
    dominant = max(scores, key=scores.get)
    mismatch = (
        (dominant in _POSITIVE_EMOTIONS and ctx_polarity < 0) or
        (dominant in _NEGATIVE_EMOTIONS and ctx_polarity > 0)
    )

    base_weight      = min(float(ctx_strength ** 0.5) * _MAX_CTX_WEIGHT, _MAX_CTX_WEIGHT)
    effective_weight = min(base_weight * 1.5, _MISMATCH_CTX_CAP) if mismatch else base_weight

    # Additive blend: (1 - w) * ml + w * target
    corrected = {
        e: (1.0 - effective_weight) * float(v) + effective_weight * target.get(e, 0.0)
        for e, v in scores.items()
    }

    total = sum(corrected.values())
    if total > 1e-9:
        corrected = {k: v / total for k, v in corrected.items()}
    else:
        corrected = dict(scores)

    return corrected, round(effective_weight, 4)


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
