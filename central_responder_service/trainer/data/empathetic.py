import os
import pickle

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

_GOEMO_TO_CDM_STATE: dict = {
    'admiration':    2,
    'amusement':     4,
    'anger':         6,
    'annoyance':     5,
    'approval':     14,
    'caring':       12,
    'confusion':    11,
    'curiosity':    10,
    'desire':        1,
    'disappointment':8,
    'disapproval':   5,
    'disgust':       6,
    'embarrassment': 8,
    'excitement':    2,
    'fear':          3,
    'gratitude':    14,
    'grief':         8,
    'joy':           1,
    'love':          1,
    'nervousness':   3,
    'optimism':     14,
    'pride':         2,
    'realization':   0,
    'relief':        9,
    'remorse':       9,
    'sadness':       8,
    'surprise':      4,
    'neutral':       0,
}


_EMP_ARCHIVE_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
_EMP_SPLIT_FILES = {
    "train":      "empatheticdialogues/train.csv",
    "validation": "empatheticdialogues/valid.csv",
    "test":       "empatheticdialogues/test.csv",
}


def _download_empathetic_csv(hf_split: str) -> list:
    """Download EmpatheticDialogues tar.gz from Facebook CDN and extract the requested split."""
    import csv
    import io
    import tarfile
    import urllib.request

    archive_path = MODEL_PATH.parent / "empatheticdialogues.tar.gz"

    if not archive_path.exists():
        try:
            logger.info(f"[EmpDialogues] Downloading archive from Facebook CDN (~28 MB)...")
            with urllib.request.urlopen(_EMP_ARCHIVE_URL, timeout=300) as resp:
                data = resp.read()
            with open(archive_path, "wb") as f:
                f.write(data)
            logger.info(f"[EmpDialogues] Archive saved → {archive_path} ({len(data)/1e6:.1f} MB)")
        except Exception as e:
            logger.warning(f"[EmpDialogues] Download failed: {e}")
            return []

    member_name = _EMP_SPLIT_FILES.get(hf_split)
    if not member_name:
        logger.warning(f"[EmpDialogues] Unknown split: {hf_split}")
        return []

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            member = tar.getmember(member_name)
            f = tar.extractfile(member)
            if f is None:
                logger.warning(f"[EmpDialogues] Could not extract {member_name}")
                return []
            text = f.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        logger.info(f"[EmpDialogues] {hf_split}: {len(rows)} rows from archive.")
        return rows
    except Exception as e:
        logger.warning(f"[EmpDialogues] Failed to read {member_name} from archive: {e}")
        archive_path.unlink(missing_ok=True)
        return []


def extract_empathetic_dialogues_features(
    vader_analyzer, bert_analyzer, goe_analyzer,
    split: str = "train",
    batch_size: int = 32,
) -> tuple:
    hf_split = {"train": "train", "val": "validation", "test": "test"}.get(split, "train")
    cap      = MAX_EMPATHETIC_SAMPLES if split == "train" else MAX_EMPATHETIC_SAMPLES // 5
    empty    = np.empty((0, FEATURE_DIM), dtype=np.float32)

    cache_path = MODEL_PATH.parent / f"empathetic_{split}_cache.pkl"
    cache_key  = f"empathetic_v2_{split}_{cap}_{FEATURE_DIM}"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                X, y, has_cdm = cached["data"]
                logger.info(f"[EmpDialogues] {split} cache hit — {len(y)} utterances.")
                return np.array(X, dtype=np.float32) if X else empty, y, has_cdm
        except Exception as e:
            logger.warning(f"[EmpDialogues] {split} cache load failed: {e} — recomputing.")

    rows_raw = _download_empathetic_csv(hf_split)
    if not rows_raw:
        logger.warning(f"EmpatheticDialogues: no data for {split} split — skipping.")
        return empty, [], []

    convs: dict = {}
    skipped_labels: Counter = Counter()
    for row in rows_raw:
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

    all_rows = []
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

    texts = [r[2] for r in all_rows]
    vader_outs, bert_outs, goe_outs = _run_parallel_batches(
        bert_analyzer, goe_analyzer, texts,
        vader_analyzer=vader_analyzer,
        batch_size=batch_size, label_prefix=f"Empathetic/{split}",
    )

    from trainer.data.meld import _meld_build_cdm
    conv_history: dict = {}
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

    if features:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"cache_key": cache_key, "data": (features, labels, has_cdm_list)}, f)
            logger.info(f"  [EmpDialogues] {split} cache saved → {cache_path}")
        except Exception as e:
            logger.warning(f"  [EmpDialogues] Could not save {split} cache: {e}")

    return (
        np.array(features, dtype=np.float32) if features else empty,
        labels,
        has_cdm_list,
    )
