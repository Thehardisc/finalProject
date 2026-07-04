"""ml/features.py — Shared derived-feature math for the 103-dim meta-learner vector."""
import math
from typing import List

from shared.constants import EMOTION_LABELS, BERT_LABELS

_EPS = 1e-9
_SHARED_LABELS = [l for l in BERT_LABELS if l in EMOTION_LABELS]


def entropy(scores: dict, keys: list) -> float:
    """Shannon entropy of a probability dict over given keys (normalized)."""
    probs = [max(scores.get(k, 0.0), 0.0) for k in keys]
    total = sum(probs) + _EPS
    probs = [p / total for p in probs]
    return float(-sum(p * math.log(p + _EPS) for p in probs))


def margin(scores: dict, keys: list) -> float:
    """Top-1 minus Top-2 confidence (decisiveness)."""
    vals = sorted([scores.get(k, 0.0) for k in keys], reverse=True)
    if len(vals) >= 2:
        return float(vals[0] - vals[1])
    return float(vals[0]) if vals else 0.0


def agreement(bert: dict, goe: dict) -> float:
    """1.0 if BERT and GoEmotions pick the same top label among shared labels, else 0.0."""
    if not _SHARED_LABELS:
        return 0.0
    bert_top = max(_SHARED_LABELS, key=lambda k: bert.get(k, 0.0))
    goe_top  = max(_SHARED_LABELS, key=lambda k: goe.get(k, 0.0))
    return 1.0 if bert_top == goe_top else 0.0


def build_derived_block(bert: dict, goe: dict, vader: dict) -> List[float]:
    """Build the 7-dim derived-features block (indices [96:103] of the FV)."""
    return [
        entropy(bert,  BERT_LABELS),
        entropy(goe,   EMOTION_LABELS),
        margin(bert,   BERT_LABELS),
        margin(goe,    EMOTION_LABELS),
        agreement(bert, goe),
        abs(float(vader.get("vader_compound", 0.0))),
        max((goe.get(k, 0.0) for k in EMOTION_LABELS), default=0.0),
    ]
