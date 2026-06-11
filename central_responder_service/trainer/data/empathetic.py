"""
trainer/data/empathetic.py — EmpatheticDialogues feature extraction with real multi-turn CDM.

Uses facebook/empathetic_dialogues (17,844 conversations, ~76k utterances) instead of the
single-sentence bdotloh variant. Each utterance gets a real CDM vector built from the
preceding turns in its conversation — same logic as meld.py — so has_cdm=True for every
turn that has at least one prior utterance.
"""

import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from shared.constants import (
    EMOTION_LABELS, FEATURE_DIM, CDM_CTX_DIM, PRIOR_DIM,
    CTX_CURR_VALENCE, CTX_MSG_LENGTH, CTX_HMM_CONF,
)
from shared.utils.logger import get_logger
from meta_learner import build_feature_vector
from trainer.utils import _run_parallel_batches

logger = get_logger("trainer")

MODEL_PATH             = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
MAX_EMPATHETIC_SAMPLES = int(os.environ.get("MAX_EMPATHETIC_SAMPLES", 25_000))

# ── EmpatheticDialogues 32 emotions → GoEmotions 28 labels ────────────────────
_EMPATHETIC_TO_GOEMOTION: dict = {
    'sentimental': 'love',         'afraid':       'fear',
    'proud':       'pride',        'faithful':     'caring',
    'terrified':   'fear',         'joyful':       'joy',
    'angry':       'anger',        'sad':          'sadness',
    'jealous':     'disapproval',  'grateful':     'gratitude',
    'embarrassed': 'embarrassment',
    'excited':     'excitement',   'annoyed':      'annoyance',
    'lonely':      'sadness',      'surprised':    'surprise',
    'furious':     'anger',        'disappointed': 'disappointment',
    'caring':      'caring',       'trusting':     'approval',
    'disgusted':   'disgust',      'anticipating': 'desire',
    'anxious':     'nervousness',  'nostalgic':    'curiosity',
    'confident':   'pride',
    'devastated':  'grief',        'hopeful':      'optimism',
    'guilty':      'remorse',      'impressed':    'admiration',
    'apprehensive':'nervousness',  'touched':      'caring',
}

# GoEmotions label → primary CDM intent state (derived from synthetic._LABEL_TO_INTENT primaries)
_GOEMO_TO_CDM_STATE: dict = {
    'admiration': 0, 'amusement': 1,   'anger': 6,       'annoyance': 6,
    'approval':   0, 'caring': 7,      'confusion': 8,   'curiosity': 8,
    'desire':     9, 'disappointment': 3, 'disapproval': 5, 'disgust': 5,
    'embarrassment': 3, 'excitement': 4, 'fear': 2,      'gratitude': 0,
    'grief':      3, 'joy': 4,         'love': 7,        'nervousness': 2,
    'optimism':   4, 'pride': 0,       'realization': 8, 'relief': 4,
    'remorse':    3, 'sadness': 3,     'surprise': 1,    'neutral': 0,
}


def extract_empathetic_dialogues_features(
    vader_analyzer, bert_analyzer, goe_analyzer,
    split: str = "train",
    batch_size: int = 32,
) -> tuple:
    """
    Build (X [N, FEATURE_DIM], y [N], has_cdm [N]) from facebook/empathetic_dialogues.

    Groups utterances by conv_id, sorts by utterance_idx, then for each utterance
    builds a CDM context vector from the preceding turns (identical logic to meld.py).
    Returns has_cdm=True for all turns with at least one prior utterance in the same
    conversation — giving real gradient signal to the context encoder.

    Capped at MAX_EMPATHETIC_SAMPLES total utterances (train) or MAX//5 (val/test).
    """
    hf_split = {"train": "train", "val": "validation", "test": "test"}.get(split, "train")
    cap      = MAX_EMPATHETIC_SAMPLES if split == "train" else MAX_EMPATHETIC_SAMPLES // 5

    try:
        from datasets import load_dataset
        emp = load_dataset("facebook/empathetic_dialogues", split=hf_split)
    except Exception as e:
        logger.warning(f"EmpatheticDialogues load failed: {e} — skipping {split} split.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), [], []

    # Group rows by conversation, preserving only mappable emotions
    convs: dict = {}
    skipped_labels: Counter = Counter()
    for row in emp:
        context     = str(row.get("context", "neutral")).lower()
        goemo_label = _EMPATHETIC_TO_GOEMOTION.get(context)
        if goemo_label is None:
            skipped_labels[context] += 1
            continue
        cid  = row["conv_id"]
        text = str(row.get("utterance", "")).strip()
        if not text:
            continue
        if cid not in convs:
            convs[cid] = {"label": goemo_label, "turns": []}
        convs[cid]["turns"].append({
            "utterance_idx": int(row.get("utterance_idx", 0)),
            "utterance":     text,
            "speaker_idx":   int(row.get("speaker_idx", 0)),
        })

    for conv in convs.values():
        conv["turns"].sort(key=lambda x: x["utterance_idx"])

    # Flatten to utterance list, capping at `cap` total
    all_rows = []  # (cid, turn_pos, text, goemo_label, speaker_idx)
    for cid, conv in convs.items():
        for t_pos, turn in enumerate(conv["turns"]):
            all_rows.append((cid, t_pos, turn["utterance"], conv["label"], turn["speaker_idx"]))
        if len(all_rows) >= cap:
            break
    all_rows = all_rows[:cap]

    n_convs = len({r[0] for r in all_rows})
    logger.info(
        f"  [EmpDialogues] {split}: {n_convs} conversations → {len(all_rows)} utterances "
        f"(skipped: {dict(skipped_labels.most_common(3))})"
    )
    if not all_rows:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), [], []

    # Batch NLP on all utterances at once
    texts = [r[2] for r in all_rows]
    vader_outs, bert_outs, goe_outs = _run_parallel_batches(
        bert_analyzer, goe_analyzer, texts,
        vader_analyzer=vader_analyzer,
        batch_size=batch_size, label_prefix=f"Empathetic/{split}",
    )

    # Build CDM vectors by replaying each conversation in order
    from trainer.data.meld import _meld_build_cdm
    conv_history: dict = {}  # cid → [{valence, intent_state, speaker, goe_dist}, ...]
    features, labels, has_cdm_list = [], [], []

    for (cid, t_pos, text, goemo_label, speaker_idx), v_out, b_out, g_out in zip(
        all_rows, vader_outs, bert_outs, goe_outs
    ):
        valence   = float(v_out.get("vader_compound", 0.0))
        intent_st = _GOEMO_TO_CDM_STATE.get(goemo_label, 0)
        history   = conv_history.get(cid, [])

        if not history:
            ctx             = np.zeros(CDM_CTX_DIM, dtype=np.float32)
            ctx[intent_st]  = 1.0
            ctx[CTX_CURR_VALENCE] = valence
            ctx[CTX_MSG_LENGTH]   = float(len(text))
            ctx[CTX_HMM_CONF]     = 0.30
            prior   = [0.0] * PRIOR_DIM
            has_cdm = False
        else:
            ctx     = _meld_build_cdm(history, valence, intent_st, speaker_idx, text)
            prior   = history[-1]["goe_dist"]
            has_cdm = True

        fv = build_feature_vector(
            {"vader": v_out, "basic_bert": b_out, "go_emotions": g_out},
            context_vector=ctx,
            trajectory_prior=prior,
        )
        features.append(fv.flatten())
        labels.append(goemo_label)
        has_cdm_list.append(has_cdm)

        goe_dist = [float(g_out.get(e, 0.0)) for e in EMOTION_LABELS]
        conv_history.setdefault(cid, []).append({
            "valence":      valence,
            "intent_state": intent_st,
            "speaker":      speaker_idx,
            "goe_dist":     goe_dist,
        })

    real_count = sum(has_cdm_list)
    logger.info(
        f"  [EmpDialogues] {split}: {len(features)} features, "
        f"{real_count} ({100 * real_count / max(len(features), 1):.1f}%) with real CDM."
    )
    empty = np.empty((0, FEATURE_DIM), dtype=np.float32)
    return (
        np.array(features, dtype=np.float32) if features else empty,
        labels,
        has_cdm_list,
    )
