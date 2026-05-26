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
import math
import statistics
import numpy as np
from collections import Counter

from shared.utils.logger import get_logger
from shared.constants import EMOTION_LABELS, VADER_KEYS, BERT_LABELS

logger = get_logger("trainer")

# Shared emotion labels between BERT and GoEmotions (used for agreement feature)
_SHARED_LABELS = [l for l in BERT_LABELS if l in EMOTION_LABELS]
_EPS = 1e-9


# ── Derived-feature helpers ───────────────────────────────────────────────────

def _entropy(scores: dict, keys: list) -> float:
    """Shannon entropy of a probability dict over given keys (normalized)."""
    probs = [max(scores.get(k, 0.0), 0.0) for k in keys]
    total = sum(probs) + _EPS
    probs = [p / total for p in probs]
    return float(-sum(p * math.log(p + _EPS) for p in probs))


def _margin(scores: dict, keys: list) -> float:
    """Top-1 minus Top-2 confidence (decisiveness)."""
    vals = sorted([scores.get(k, 0.0) for k in keys], reverse=True)
    if len(vals) >= 2:
        return float(vals[0] - vals[1])
    return float(vals[0]) if vals else 0.0


def _agreement(bert: dict, goe: dict) -> float:
    """1.0 if BERT and GoEmotions agree on the top shared-label class, else 0.0."""
    if not _SHARED_LABELS:
        return 0.0
    bert_top = max(_SHARED_LABELS, key=lambda k: bert.get(k, 0.0))
    goe_top  = max(_SHARED_LABELS, key=lambda k: goe.get(k, 0.0))
    return 1.0 if bert_top == goe_top else 0.0


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

    # Block 6: derived features (7)
    vec.append(_entropy(bert,  BERT_LABELS))           # how uncertain is BERT?
    vec.append(_entropy(goe,   EMOTION_LABELS))        # how uncertain is GoEmotions?
    vec.append(_margin(bert,   BERT_LABELS))           # how decisive is BERT?
    vec.append(_margin(goe,    EMOTION_LABELS))        # how decisive is GoEmotions?
    vec.append(_agreement(bert, goe))                  # do both transformers agree?
    vec.append(abs(float(vader.get("vader_compound", 0.0))))  # sentiment magnitude
    vec.append(max((goe.get(k, 0.0) for k in EMOTION_LABELS), default=0.0))  # GoE peak

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
