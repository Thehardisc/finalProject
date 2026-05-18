"""
trainer/preprocessor.py — Feature vector construction and dataset filters.
"""
import statistics
import numpy as np
from collections import Counter

from shared.utils.logger import get_logger
from shared.constants import EMOTION_LABELS, VADER_KEYS, BERT_LABELS

logger = get_logger("trainer")


def build_fv(vader: dict, bert: dict, goe: dict, emoji: dict,
             context: dict = None) -> np.ndarray:
    """Assemble the 96-dimensional feature vector from 4 model score dicts + context."""
    context = context or {}
    vec = []
    for k in VADER_KEYS:     vec.append(float(vader.get(k, 0.0)))
    for k in BERT_LABELS:    vec.append(float(bert.get(k,  0.0)))
    for k in EMOTION_LABELS: vec.append(float(goe.get(k,   0.0)))
    for k in EMOTION_LABELS: vec.append(float(emoji.get(k,  0.0)))

    # Context block
    vec.append(float(context.get("avg_valence", 0.0)))
    prev_emo = context.get("prev_emotion", "neutral").lower()
    for label in EMOTION_LABELS:
        vec.append(1.0 if label == prev_emo else 0.0)

    return np.array(vec, dtype=np.float32)


def filter_outliers(X, y, goe_list):
    """Drop samples where GoEmotions gives <5% confidence to the gold label."""
    cX, cy, removed = [], [], 0
    for fv, label, goe in zip(X, y, goe_list):
        if label not in EMOTION_LABELS or goe.get(label, 0.0) < 0.05:
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
