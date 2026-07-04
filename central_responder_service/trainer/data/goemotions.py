import os
import pickle
import time
from pathlib import Path

import numpy as np

from shared.constants import EMOTION_LABELS, FEATURE_DIM
from shared.utils.logger import get_logger
from meta_learner import build_feature_vector
from trainer.utils import _run_batch, _run_parallel_batches, _vader

logger = get_logger("trainer")

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))

_GOE_DIRECT_PER_CLASS   = 400
_GOE_DIRECT_CACHE_ID    = "goemotions_direct_v4"


def extract_goemotions_direct_features(
    vader_analyzer, bert_analyzer, goe_analyzer,
    batch_size: int = 32,
) -> tuple:
    from trainer.logging_utils import _StderrToLogger

    cache_path = MODEL_PATH.parent / "goemotions_direct_cache.pkl"
    cache_key  = f"{_GOE_DIRECT_CACHE_ID}_{_GOE_DIRECT_PER_CLASS}_{FEATURE_DIM}"
    empty      = np.empty((0, FEATURE_DIM), dtype=np.float32)

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                X, y, gs = cached["data"]
                logger.info(
                    f"[GoEDirect] Feature cache hit — loaded {len(y)} NLP-aligned samples."
                )
                return np.array(X, dtype=np.float32) if X else empty, y, gs
            logger.info("[GoEDirect] Cache stale — recomputing.")
        except Exception as e:
            logger.warning(f"[GoEDirect] Cache load failed: {e} — recomputing.")

    logger.info(
        f"[GoEDirect] Loading GoEmotions train+val+test (≤{_GOE_DIRECT_PER_CLASS}/class × 28 classes)..."
    )
    try:
        from datasets import load_dataset
        with _StderrToLogger():
            logger.info("[GoEDirect] Downloading split=train...")
            ds_train = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
            logger.info(f"[GoEDirect] train loaded ({len(ds_train)} rows). Downloading split=validation...")
            ds_val   = load_dataset("google-research-datasets/go_emotions", "simplified", split="validation")
            logger.info(f"[GoEDirect] validation loaded ({len(ds_val)} rows). Downloading split=test...")
            ds_test  = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
            logger.info(f"[GoEDirect] test loaded ({len(ds_test)} rows).")
    except Exception as e:
        logger.warning(f"[GoEDirect] Could not load GoEmotions dataset: {e} — skipping.")
        return empty, [], []

    id2label = ds_val.features["labels"].feature.int2str

    per_class: dict = {lbl: [] for lbl in EMOTION_LABELS}
    for row in list(ds_train) + list(ds_val) + list(ds_test):
        if len(row["labels"]) != 1:
            continue
        label_str = id2label(row["labels"][0])
        if label_str in per_class and len(per_class[label_str]) < _GOE_DIRECT_PER_CLASS:
            per_class[label_str].append(row["text"])

    counts = {k: len(v) for k, v in per_class.items() if v}
    total  = sum(counts.values())
    logger.info(f"[GoEDirect] Collected {total} samples across {len(counts)} classes: {counts}")

    if total == 0:
        return empty, [], []

    pairs = [(text, lbl) for lbl, texts in per_class.items() for text in texts]
    all_texts = [t for t, _ in pairs]
    logger.info(f"[GoEDirect] Running batched NLP on {len(all_texts)} texts...")
    t0 = time.time()

    vader_outs = []
    if callable(bert_analyzer) and callable(goe_analyzer):
        vader_outs, bert_outs, goe_outs = _run_parallel_batches(
            bert_analyzer, goe_analyzer, all_texts,
            vader_analyzer=vader_analyzer,
            batch_size=batch_size, label_prefix="GoEDirect",
        )
    else:
        for text in all_texts:
            try:
                vader_outs.append({f"vader_{k}": v for k, v in _vader(vader_analyzer, text).items()})
            except Exception:
                vader_outs.append({})
        bert_outs = _run_batch(bert_analyzer, all_texts, batch_size=batch_size, label="GoEDirect/BERT") if callable(bert_analyzer) else [{} for _ in all_texts]
        goe_outs  = _run_batch(goe_analyzer,  all_texts, batch_size=batch_size, label="GoEDirect/GoE")  if callable(goe_analyzer)  else [{} for _ in all_texts]

    elapsed = time.time() - t0
    logger.info(
        f"[GoEDirect] NLP done — {len(all_texts)} texts in {elapsed:.0f}s "
        f"({len(all_texts)/elapsed:.1f} samples/s)"
    )

    features, labels, gs_list = [], [], []
    for (_, lbl), vader_out, bert_out, goe_out in zip(pairs, vader_outs, bert_outs, goe_outs):
        fv = build_feature_vector(
            {"vader": vader_out, "basic_bert": bert_out, "go_emotions": goe_out},
        )
        features.append(fv.flatten())
        labels.append(lbl)
        gs_list.append(goe_out)

    logger.info(f"[GoEDirect] Built {len(features)} NLP-aligned feature vectors.")

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"cache_key": cache_key, "data": (features, labels, gs_list)}, f)
        logger.info(f"[GoEDirect] Cache saved → {cache_path}")
    except Exception as e:
        logger.warning(f"[GoEDirect] Could not save cache: {e}")

    return np.array(features, dtype=np.float32) if features else empty, labels, gs_list
