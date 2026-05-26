"""
ml/predictor.py — Feature vector construction and meta-learner inference.
"""
import math
import numpy as np
from typing import Tuple, Optional

from shared.utils.logger import get_logger
from shared.constants import EMOTION_LABELS, VADER_KEYS, BERT_LABELS

from .conflict_detector import detect_emotional_conflicts

logger = get_logger("meta_learner")

# Shared emotion labels between BERT and GoEmotions (used for agreement feature)
_SHARED_LABELS = [l for l in BERT_LABELS if l in EMOTION_LABELS]
_EPS = 1e-9


def _entropy(scores: dict, keys: list) -> float:
    probs = [max(scores.get(k, 0.0), 0.0) for k in keys]
    total = sum(probs) + _EPS
    probs = [p / total for p in probs]
    return float(-sum(p * math.log(p + _EPS) for p in probs))


def _margin(scores: dict, keys: list) -> float:
    vals = sorted([scores.get(k, 0.0) for k in keys], reverse=True)
    return float(vals[0] - vals[1]) if len(vals) >= 2 else (float(vals[0]) if vals else 0.0)


def _agreement(bert: dict, goe: dict) -> float:
    if not _SHARED_LABELS:
        return 0.0
    bert_top = max(_SHARED_LABELS, key=lambda k: bert.get(k, 0.0))
    goe_top  = max(_SHARED_LABELS, key=lambda k: goe.get(k, 0.0))
    return 1.0 if bert_top == goe_top else 0.0


def build_feature_vector(model_outputs: dict, context: dict = None) -> np.ndarray:
    """
    Build a fixed-length float32 numpy feature vector from 4 model output dicts
    plus conversation context.

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

    # Block 5: Context (29)
    vec.append(float(context.get("avg_valence", 0.0)))
    prev_emo = context.get("prev_emotion", "neutral").lower()
    for label in EMOTION_LABELS:
        vec.append(1.0 if label == prev_emo else 0.0)

    # Block 6: Derived features (7) — MUST match trainer/preprocessor.py exactly
    vec.append(_entropy(bert_scores,       BERT_LABELS))    # bert_entropy
    vec.append(_entropy(goemotions_scores, EMOTION_LABELS)) # goe_entropy
    vec.append(_margin(bert_scores,        BERT_LABELS))    # bert_margin
    vec.append(_margin(goemotions_scores,  EMOTION_LABELS)) # goe_margin
    vec.append(_agreement(bert_scores, goemotions_scores))  # bert_goe_agreement
    vec.append(abs(float(vader_scores.get("vader_compound", 0.0))))  # vader_abs_compound
    vec.append(max((goemotions_scores.get(k, 0.0) for k in EMOTION_LABELS), default=0.0))  # max_goe

    arr = np.array(vec, dtype=np.float32)
    return arr.reshape(1, -1)


def predict_with_meta_learner(
    model, feature_vector: np.ndarray
) -> Tuple[str, float, dict, float, Optional[str]]:
    """
    Run inference with the meta-learner.

    Returns:
        (dominant_emotion, confidence, all_scores, sarcasm_score, conflict_desc)
        On error returns ("neutral", 0.0, {}, 0.0, None) — never raises.
    """
    try:
        pred_label = model.predict(feature_vector)[0]
        proba      = model.predict_proba(feature_vector)[0]
        classes    = model.classes_

        all_scores = {str(k): float(v) for k, v in zip(classes, proba)}
        label_idx  = list(classes).index(pred_label)
        confidence = float(proba[label_idx])

        sarcasm_score, conflict_desc = detect_emotional_conflicts(feature_vector)

        # Explicit sarcasm override — flip label to negative valence
        if sarcasm_score > 0.4:
            if conflict_desc and ("disapproval" in conflict_desc.lower()
                                  or "passive-aggression" in conflict_desc.lower()):
                pred_label = "disapproval"
            else:
                pred_label = "annoyance"
            confidence = max(confidence, sarcasm_score)
            all_scores[pred_label] = max(all_scores.get(pred_label, 0.0), sarcasm_score)

        return pred_label, confidence, all_scores, sarcasm_score, conflict_desc

    except Exception as e:
        logger.warning(f"Predict error: {e}. Returning neutral.")
        return "neutral", 0.0, {}, 0.0, None
