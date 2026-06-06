"""
trainer/data/empathetic.py — EmpatheticDialogues feature extraction + HMM utils.
"""

import os
import time
from pathlib import Path

import numpy as np

from shared.constants import (
    EMOTION_LABELS, FEATURE_DIM, CDM_CTX_DIM, PRIOR_DIM,
)
from shared.utils.logger import get_logger
from meta_learner import build_feature_vector
from trainer.utils import _run_batch, _vader
from trainer.data.synthetic import build_synthetic_context_vector

logger = get_logger("trainer")

MODEL_PATH              = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
MAX_EMPATHETIC_SAMPLES  = int(os.environ.get("MAX_EMPATHETIC_SAMPLES", 25_000))

# ── EmpatheticDialogues conversation-context training data ─────────────────────

# Map EmpatheticDialogues 32 emotions → GoEmotions 28 labels
_EMPATHETIC_TO_GOEMOTION: dict = {
    'sentimental': 'love',       'afraid':       'fear',
    'proud':       'pride',      'faithful':     'caring',
    'terrified':   'fear',       'joyful':       'joy',
    'angry':       'anger',      'sad':          'sadness',
    'jealous':     'disapproval','grateful':     'gratitude',
    'embarrassed':  'embarrassment',
    'excited':     'excitement', 'annoyed':      'annoyance',
    'lonely':      'sadness',    'surprised':    'surprise',
    'furious':     'anger',      'disappointed': 'disappointment',
    'caring':      'caring',     'trusting':     'approval',
    'disgusted':   'disgust',    'anticipating': 'desire',
    'anxious':     'nervousness','nostalgic':    'curiosity',
    'confident':   'pride',
    'devastated':  'grief',      'hopeful':      'optimism',
    'guilty':      'remorse',    'impressed':    'admiration',
    'apprehensive':'nervousness','touched':      'caring',
}

# Rough valence for EmpatheticDialogues emotions (used to approximate velocity)
_EMPATHETIC_VALENCE: dict = {
    'joyful': 0.82, 'excited': 0.78, 'grateful': 0.72, 'proud': 0.68,
    'content': 0.55, 'hopeful': 0.60, 'anticipating': 0.40, 'trusting': 0.50,
    'caring': 0.58, 'faithful': 0.50, 'sentimental': 0.40, 'impressed': 0.65,
    'touched': 0.55, 'prepared': 0.35, 'nostalgic': 0.10, 'confident': 0.60,
    'surprised': 0.15,
    'angry': -0.72, 'furious': -0.85, 'sad': -0.68, 'terrified': -0.80,
    'afraid': -0.62, 'lonely': -0.65, 'disappointed': -0.58, 'annoyed': -0.48,
    'embarrassed': -0.42, 'disgusted': -0.70, 'devastated': -0.88,
    'jealous': -0.52, 'guilty': -0.58, 'anxious': -0.45, 'apprehensive': -0.40,
}


def _load_hmm_params():
    """Load HMM transmat and emissionprob from models/cdm_hmm.pkl."""
    hmm_path = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")).parent / "cdm_hmm.pkl"
    if not hmm_path.exists():
        return None, None
    try:
        import pickle as pkl
        with open(hmm_path, "rb") as f:
            d = pkl.load(f)
        return np.array(d["transmat"]), np.array(d["emissionprob"])
    except Exception as e:
        logger.warning(f"Could not load CDM HMM for EmpatheticDialogues: {e}")
        return None, None


def _hmm_forward_step(alpha, transmat, emissionprob, obs):
    """Single HMM forward step: α_t = normalise((α_{t-1} @ A) ⊙ B[:,obs])."""
    pred = alpha @ transmat
    upd  = pred * emissionprob[:, obs]
    s    = upd.sum()
    return upd / s if s > 1e-12 else np.ones(len(alpha)) / len(alpha)


def _empathetic_obs(emotion_label: str) -> int:
    """Map an EmpatheticDialogues emotion to a DailyDialog-compatible obs index."""
    _EMO_TO_7 = {
        'joy': 4, 'joyful': 4, 'excited': 4, 'grateful': 4, 'proud': 4,
        'content': 4, 'hopeful': 4, 'trusting': 4, 'caring': 4, 'impressed': 4,
        'anger': 1, 'angry': 1, 'furious': 1, 'annoyed': 1, 'jealous': 1,
        'disgust': 2, 'disgusted': 2, 'guilty': 2,
        'fear': 3, 'afraid': 3, 'terrified': 3, 'anxious': 3, 'apprehensive': 3,
        'sadness': 5, 'sad': 5, 'lonely': 5, 'devastated': 5, 'disappointed': 5,
        'embarrassed': 5, 'sentimental': 5,
        'surprise': 6, 'surprised': 6,
    }
    emot_idx = _EMO_TO_7.get(emotion_label.lower(), 0)
    val = _EMPATHETIC_VALENCE.get(emotion_label.lower(), 0.0)
    act_idx  = 3 if val > 0.3 else (1 if val > -0.3 else 2)
    return act_idx * 7 + emot_idx


def extract_empathetic_dialogues_features(
    vader_analyzer, bert_analyzer, goe_analyzer,
    split: str = "train",
    batch_size: int = 32,
) -> tuple:
    """
    Build (X [N, FEATURE_DIM], y [N], gs [N]) from EmpatheticDialogues.

    The dataset's native splits are used directly (train ~19k / val ~2.7k / test ~2.5k).
    Size is capped by MAX_EMPATHETIC_SAMPLES (train) and MAX_EMPATHETIC_SAMPLES // 5 (val/test).

    Labels: 32 EmpatheticDialogues emotions → GoEmotions 28 via _EMPATHETIC_TO_GOEMOTION.
    CDM context: synthetic (has_cdm=False — single-turn data, no conversation history).
    ctx_mode: "train" for train split, "val" for val, "cold" for test.
    """
    hf_split = {"train": "train", "val": "validation", "test": "test"}.get(split, "train")
    cap      = MAX_EMPATHETIC_SAMPLES if split == "train" else MAX_EMPATHETIC_SAMPLES // 5
    ctx_mode = {"train": "train", "val": "val", "test": "cold"}[split]

    try:
        from datasets import load_dataset
        emp = load_dataset("bdotloh/empathetic-dialogues-contexts", split=hf_split)
    except Exception as e:
        logger.warning(f"EmpatheticDialogues load failed: {e} — skipping {split} split.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), [], []

    rows = list(emp)[:cap]
    total = len(rows)

    # Pass 1: filter rows and collect texts (cheap — no model calls)
    texts, goemo_labels, contexts = [], [], []
    for row in rows:
        text        = str(row.get('situation', '')).strip()
        raw_emotion = str(row.get('emotion', 'neutral')).lower()
        goemo_label = _EMPATHETIC_TO_GOEMOTION.get(raw_emotion)
        if not text or goemo_label is None:
            continue
        texts.append(text)
        goemo_labels.append(goemo_label)
        contexts.append(build_synthetic_context_vector(label=goemo_label, mode=ctx_mode))

    mapped  = len(texts)
    skipped = total - mapped
    logger.info(
        f"  [EmpDialogues] {split}: {total} rows loaded, "
        f"{mapped} mapped ({skipped} skipped — unmapped emotions: prepared/content/ashamed)"
    )

    if not texts:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), [], []

    # Pass 2: batch NLP inference (the expensive step)
    t0 = time.time()
    vader_outs = [{f"vader_{k}": v for k, v in _vader(vader_analyzer, t).items()} for t in texts]
    bert_outs  = _run_batch(bert_analyzer, texts, batch_size=batch_size, label=f"Empathetic/{split}/BERT")
    goe_outs   = _run_batch(goe_analyzer,  texts, batch_size=batch_size, label=f"Empathetic/{split}/GoE")

    # Pass 3: assemble feature vectors (cheap — no model calls)
    features, labels, gs_list = [], [], []
    for goemo_label, v_out, b_out, g_out, ctx in zip(goemo_labels, vader_outs, bert_outs, goe_outs, contexts):
        fv = build_feature_vector(
            {"vader": v_out, "basic_bert": b_out, "go_emotions": g_out},
            context_vector=ctx[:CDM_CTX_DIM],
            trajectory_prior=ctx[CDM_CTX_DIM:],
        )
        features.append(fv.flatten())
        labels.append(goemo_label)
        gs_list.append(g_out)

    elapsed = time.time() - t0
    logger.info(
        f"  [EmpDialogues] {len(features)} samples ({split}) "
        f"in {elapsed:.0f}s ({len(features)/elapsed:.1f} samples/s)."
    )
    empty = np.empty((0, FEATURE_DIM), dtype=np.float32)
    return (
        np.array(features, dtype=np.float32) if features else empty,
        labels,
        gs_list,
    )
