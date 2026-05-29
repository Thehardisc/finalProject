"""
trainer/preprocessor.py — Feature vector construction and dataset filters.

FEATURE_DIM: 96 base + 7 derived = 103 total
  Block 1: VADER         (4)
  Block 2: BERT Ekman    (7)
  Block 3: GoEmotions    (28)
  Block 4: EmojiNet      (28)
  Block 5: Context       (29)  valence(1) + one-hot prev emotion(28)
  Block 6: Derived       (7)   entropy×2, margin×2, agreement, |compound|, max_goe
"""
import statistics
import numpy as np
from collections import Counter

from shared.utils.logger import get_logger
from shared.constants import EMOTION_LABELS, VADER_KEYS, BERT_LABELS

from ml.features import build_derived_block

logger = get_logger("trainer")


# ── Public API ────────────────────────────────────────────────────────────────

def build_fv(vader: dict, bert: dict, goe: dict, emoji: dict,
             context: dict = None) -> np.ndarray:
    """
    Assemble the 103-dimensional feature vector from 4 model score dicts + context.

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
    vec = []

    # Block 1-4: raw probability scores
    for k in VADER_KEYS:     vec.append(float(vader.get(k, 0.0)))
    for k in BERT_LABELS:    vec.append(float(bert.get(k,  0.0)))
    for k in EMOTION_LABELS: vec.append(float(goe.get(k,   0.0)))
    for k in EMOTION_LABELS: vec.append(float(emoji.get(k, 0.0)))

    # Block 5: context
    vec.append(float(context.get("avg_valence", 0.0)))
    prev_emo = context.get("prev_emotion", "neutral").lower()
    for label in EMOTION_LABELS:
        vec.append(1.0 if label == prev_emo else 0.0)

    # Block 6: derived features (7) — shared implementation in ml/features.py
    vec.extend(build_derived_block(bert, goe, vader))

    return np.array(vec, dtype=np.float32)


def filter_outliers(X, y, goe_list):
    """
    Drop samples where GoEmotions gives <15% confidence to the gold label.
    Raised from 5% to reduce noisy training labels.
    """
    cX, cy, removed = [], [], 0
    for fv, label, goe in zip(X, y, goe_list):
        if label not in EMOTION_LABELS or goe.get(label, 0.0) < 0.15:
            removed += 1
        else:
            cX.append(fv); cy.append(label)
    return cX, cy, removed


def filter_balance(X, y):
    """Cap any class at 3× the median class count to reduce bias."""
    if not y:
        return X, y
    counts = Counter(y)
    cap    = max(50, int(statistics.median(counts.values()) * 3))
    seen   = Counter()
    cX, cy = [], []
    for fv, label in zip(X, y):
        if seen[label] < cap:
            cX.append(fv); cy.append(label); seen[label] += 1
    removed = len(X) - len(cX)
    if removed:
        logger.info(f"  [Filter] Balance cap: removed {removed} samples (cap={cap}/class).")
    return cX, cy
