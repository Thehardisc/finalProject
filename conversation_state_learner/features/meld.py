"""conversation_state_learner/features/meld.py"""

import logging
from typing import Callable, Dict, List, Tuple

import numpy as np

from features.schema import (
    EMOTION_LABELS_28,
    BERT_LABELS_7,
    VADER_KEYS_4,
    CDM_CTX_DIM,
    MSG_DIM,
    N_EMOTIONS,
)

logger = logging.getLogger("meld")


_MELD_TO_GOEMOTION: Dict[str, str] = {
    "anger":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "joy",
    "neutral":  "neutral",
    "sadness":  "sadness",
    "surprise": "surprise",
}

_MELD_VALENCE: Dict[str, float] = {
    "joy":      0.80,
    "surprise": 0.15,
    "neutral":  0.00,
    "anger":   -0.75,
    "disgust": -0.70,
    "fear":    -0.65,
    "sadness": -0.70,
}


# ── Helper: build one 79-dim feature vector ────────────────────────────────────

def _utterance_to_vec(
    text: str,
    meld_emotion: str,
    vader_fn: Callable,
    bert_fn:  Callable,
    goe_fn:   Callable,
) -> np.ndarray:
    """Run NLP models on a single MELD utterance and return a MSG_DIM-dim vector."""
    try:
        vader_out = vader_fn(text)
        bert_out  = bert_fn(text)
        goe_out   = goe_fn(text)
    except Exception as e:
        logger.debug(f"Model error on '{text[:40]}': {e}")
        return np.zeros(MSG_DIM, dtype=np.float32)

    go_vec    = np.array([float(goe_out.get(e,  0.0)) for e in EMOTION_LABELS_28], dtype=np.float32)
    bert_vec  = np.array([float(bert_out.get(e, 0.0)) for e in BERT_LABELS_7],     dtype=np.float32)
    vader_vec = np.array([float(vader_out.get(k, 0.0)) for k in VADER_KEYS_4],     dtype=np.float32)
    cdm_vec   = np.zeros(CDM_CTX_DIM, dtype=np.float32)

    return np.concatenate([go_vec, bert_vec, vader_vec, cdm_vec])


# ── Public API ─────────────────────────────────────────────────────────────────

def load_meld_sequences(
    vader_fn:      Callable,
    bert_fn:       Callable,
    goe_fn:        Callable,
    max_dialogues: int = 1000,
    splits:        Tuple[str, ...] = ("train", "validation"),
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load MELD and build LSTM training sequences."""
    try:
        from datasets import load_dataset
        meld = load_dataset("declare-lab/MELD")
    except Exception as e:
        logger.warning(f"MELD load failed: {e} — skipping.")
        return [], []

    dialogues: Dict[int, List[dict]] = {}
    for split in splits:
        if split not in meld:
            continue
        for row in meld[split]:
            did = row["Dialogue_ID"]
            dialogues.setdefault(did, []).append(row)

    dialogue_list = list(dialogues.items())[:max_dialogues]
    logger.info(f"  [MELD] Processing {len(dialogue_list)} dialogues from {splits}...")

    sequences: List[np.ndarray] = []
    targets:   List[np.ndarray] = []
    skipped = 0

    for idx, (did, turns) in enumerate(dialogue_list):
        if idx % 100 == 0:
            logger.info(f"  [MELD] {idx}/{len(dialogue_list)} dialogues processed ({len(sequences)} valid)")

        turns_sorted = sorted(turns, key=lambda t: t["Utterance_ID"])

        vecs = []
        for turn in turns_sorted:
            text = str(turn.get("Utterance", "")).strip()
            if not text:
                continue
            vec = _utterance_to_vec(text, turn.get("Emotion", "neutral"), vader_fn, bert_fn, goe_fn)
            vecs.append(vec)

        if len(vecs) < 2:
            skipped += 1
            continue

        X = np.stack(vecs)
        y = X[1:, :N_EMOTIONS]

        sequences.append(X)
        targets.append(y)

    logger.info(
        f"  [MELD] Done: {len(sequences)} sequences built "
        f"({skipped} skipped, avg len {np.mean([len(s) for s in sequences]):.1f} turns)"
    )
    return sequences, targets


def load_meld_flat(
    vader_fn:      Callable,
    bert_fn:       Callable,
    goe_fn:        Callable,
    max_dialogues: int = 1000,
    splits:        Tuple[str, ...] = ("train",),
) -> Tuple[np.ndarray, List[str]]:
    """Load MELD as a flat feature matrix for meta-learner training."""
    try:
        from datasets import load_dataset
        meld = load_dataset("declare-lab/MELD")
    except Exception as e:
        logger.warning(f"MELD load failed: {e} — skipping.")
        return np.empty((0, MSG_DIM), dtype=np.float32), []

    rows_X, rows_y = [], []
    total = 0

    for split in splits:
        if split not in meld:
            continue
        split_data = list(meld[split])
        utterance_cap = max_dialogues * 10
        for row in split_data[:utterance_cap]:
            text = str(row.get("Utterance", "")).strip()
            if not text:
                continue
            goemo_label = _MELD_TO_GOEMOTION.get(row.get("Emotion", "neutral").lower(), "neutral")
            vec = _utterance_to_vec(text, goemo_label, vader_fn, bert_fn, goe_fn)
            rows_X.append(vec)
            rows_y.append(goemo_label)
            total += 1

    logger.info(f"  [MELD flat] {total} utterances extracted from {splits}")
    X = np.stack(rows_X) if rows_X else np.empty((0, MSG_DIM), dtype=np.float32)
    return X, rows_y
