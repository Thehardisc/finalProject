"""
ml/features.py — Shared derived-feature math for the 118-dim meta-learner vector.

The training pipeline (`trainer/preprocessor.py:build_fv`) and the inference
pipeline (`ml/predictor.py:build_feature_vector`) BOTH must produce identical
118-dim vectors for the same input. Block 6 (the 7 derived features) used to
be implemented twice — once in each file — and could silently desync.

All derived-feature math now lives here and is imported by both call sites.
"""
import math
from typing import List, Optional

import numpy as np

from shared.constants import (
    EMOTION_LABELS, BERT_LABELS, CONTEXT_DIM, CDM_N_STATES,
    CTX_CURR_VALENCE, CTX_HIST_VALENCE,
)

_EPS = 1e-9
_SHARED_LABELS = [l for l in BERT_LABELS if l in EMOTION_LABELS]


def build_cdm_context_block(context: dict) -> List[float]:
    """
    Build the 44-dim CDM context block (indices [67:111] of the FV).

    The block is produced upstream by context_engine_service's Parallel Belief
    State Machine and arrives in `context["cdm_context"]` as a 44-float list:
      [0:28]  = CDM emotion belief distribution (sums to 1)
      [28]    = top-1 state residency
      [29:32] = recent transition trace
      [32:44] = 12 CDM scalars (abruptness, coherence, entropy, valence, …)

    Fallback (context_engine missed the aggregation window): a UNIFORM belief
    (1/28 per state) with all scalars zeroed.  A uniform belief is the
    maximum-entropy "I don't know" prior — it keeps the block non-zero (so the
    meta-learner's context attribution never silently collapses to 0%) without
    asserting any particular emotional state.
    """
    cdm = context.get("cdm_context")
    if isinstance(cdm, (list, tuple)) and len(cdm) == CONTEXT_DIM:
        return [float(v) for v in cdm]

    block = [0.0] * CONTEXT_DIM
    uniform = 1.0 / CDM_N_STATES
    for i in range(CDM_N_STATES):
        block[i] = uniform
    return block


def synthesize_cdm_context_block(
    prev_emotion: Optional[str],
    avg_valence: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """
    Build a *synthetic* 44-dim CDM context block for TRAINING.

    At serve time the belief block is a soft PBSM distribution (peaked, but never
    one-hot).  Training on a hard one-hot prev-emotion would create train/serve
    skew — the model would learn to trust razor-sharp context that it never
    actually receives.  Instead we draw a Dirichlet belief concentrated on the
    previous dominant emotion, mirroring the PBSM's output shape, and fill the
    valence scalars from the conversation's running average.

    The remaining CDM scalars are left at 0.0: in training we have no live
    transition/coherence telemetry, and zeroing them keeps the synthetic
    distribution honest rather than inventing signal the model could overfit to.
    """
    rng = rng or np.random.default_rng()
    block = [0.0] * CONTEXT_DIM

    # Dirichlet belief peaked at prev_emotion (soft, sums to 1)
    alpha = np.full(CDM_N_STATES, 0.5)
    prev = (prev_emotion or "neutral").lower()
    if prev in EMOTION_LABELS:
        alpha[EMOTION_LABELS.index(prev)] += 4.0
    belief = rng.dirichlet(alpha)
    block[:CDM_N_STATES] = belief.tolist()

    # Valence scalars (the only context signal available offline)
    v = float(np.clip(avg_valence, -1.0, 1.0))
    block[CTX_CURR_VALENCE] = v
    block[CTX_HIST_VALENCE] = v

    return block


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
    """
    Build the 7-dim derived-features block (indices [111:118] of the FV).

    Layout:
      0: bert_entropy        — how uncertain is BERT?
      1: goe_entropy         — how uncertain is GoEmotions?
      2: bert_margin         — how decisive is BERT?
      3: goe_margin          — how decisive is GoEmotions?
      4: bert_goe_agreement  — do both transformers pick the same top label?
      5: vader_abs_compound  — sentiment magnitude
      6: max_goe_score       — GoEmotions peak confidence
    """
    return [
        entropy(bert,  BERT_LABELS),
        entropy(goe,   EMOTION_LABELS),
        margin(bert,   BERT_LABELS),
        margin(goe,    EMOTION_LABELS),
        agreement(bert, goe),
        abs(float(vader.get("vader_compound", 0.0))),
        max((goe.get(k, 0.0) for k in EMOTION_LABELS), default=0.0),
    ]
