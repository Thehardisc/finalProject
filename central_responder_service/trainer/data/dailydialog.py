"""
dair-ai/emotion dataset extractor.
416K English short texts labeled with 6 emotions (sadness, joy, love, anger, fear, surprise).
Available as parquet — no dataset script required.
"""
import gc
import os
import pickle
import time
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

MODEL_PATH           = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
MAX_DAIREMO_SAMPLES  = int(os.environ.get("MAX_DAIREMO_SAMPLES", 20_000))

_DAIREMO_TO_GOEMOTION: dict = {
    "sadness":  "sadness",
    "joy":      "joy",
    "love":     "love",
    "anger":    "anger",
    "fear":     "fear",
    "surprise": "surprise",
}

_DAIREMO_VALENCE: dict = {
    "sadness":  -0.70,
    "joy":       0.80,
    "love":      0.75,
    "anger":    -0.70,
    "fear":     -0.60,
    "surprise":  0.20,
}

_LABEL_MAP: dict = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}


def extract_dailydialog_features(
    vader_analyzer,
    bert_analyzer,
    goe_analyzer,
    max_utterances: int = MAX_DAIREMO_SAMPLES,
    batch_size: int = 32,
) -> tuple:
    """Extract 116-dim feature vectors from dair-ai/emotion (parquet, no CDM context)."""
    cache_path = MODEL_PATH.parent / "dairemo_features_cache.pkl"
    cache_key  = f"dairemo_v1_{max_utterances}_{FEATURE_DIM}"
    empty      = np.empty((0, FEATURE_DIM), dtype=np.float32)

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                X, y, has_cdm = cached["data"]
                logger.info(f"[DairEmo] Cache hit — {len(y)} samples.")
                return np.array(X, dtype=np.float32) if X else empty, y, has_cdm
        except Exception as e:
            logger.warning(f"[DairEmo] Cache load failed: {e} — recomputing.")

    try:
        from datasets import load_dataset
        logger.info("[DairEmo] Loading dair-ai/emotion (parquet, train split)...")
        ds = load_dataset("dair-ai/emotion", split="train")
    except Exception as e:
        logger.warning(f"[DairEmo] Load failed: {e} — skipping.")
        return empty, [], []

    all_rows = []
    for item in ds:
        text  = str(item.get("text", "")).strip()
        label_id = int(item.get("label", 0))
        if not text:
            continue
        goemo_label = _LABEL_MAP.get(label_id)
        if goemo_label not in _DAIREMO_TO_GOEMOTION:
            continue
        all_rows.append((text, _DAIREMO_TO_GOEMOTION[goemo_label]))
        if len(all_rows) >= max_utterances:
            break

    del ds
    gc.collect()

    if not all_rows:
        logger.warning("[DairEmo] No rows extracted — skipping.")
        return empty, [], []

    logger.info(f"[DairEmo] {len(all_rows)} samples loaded. Running NLP...")
    texts  = [r[0] for r in all_rows]
    labels = [r[1] for r in all_rows]

    t0 = time.time()
    vader_outs, bert_outs, goe_outs = _run_parallel_batches(
        bert_analyzer, goe_analyzer, texts,
        vader_analyzer=vader_analyzer,
        batch_size=batch_size, label_prefix="DairEmo",
    )
    logger.info(f"[DairEmo] NLP done in {time.time()-t0:.0f}s. Building feature vectors...")

    features, has_cdm_list = [], []
    for (text, goemo_label), v_out, b_out, g_out in zip(all_rows, vader_outs, bert_outs, goe_outs):
        ctx = np.zeros(CDM_CTX_DIM, dtype=np.float32)
        ctx[CTX_CURR_VALENCE] = float(v_out.get("vader_compound", _DAIREMO_VALENCE.get(goemo_label, 0.0)))
        ctx[CTX_MSG_LENGTH]   = float(len(text))
        ctx[CTX_HMM_CONF]     = 0.25
        prior = [0.0] * PRIOR_DIM

        fv = build_feature_vector(
            {"vader": v_out, "basic_bert": b_out, "go_emotions": g_out},
            context_vector=ctx,
            trajectory_prior=prior,
        )
        features.append(fv.flatten())
        has_cdm_list.append(False)

    logger.info(f"[DairEmo] {len(features)} feature vectors built.")

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"cache_key": cache_key, "data": (features, labels, has_cdm_list)}, f)
        logger.info(f"[DairEmo] Cache saved → {cache_path}")
    except Exception as e:
        logger.warning(f"[DairEmo] Could not save cache: {e}")

    return np.array(features, dtype=np.float32) if features else empty, labels, has_cdm_list
