import csv
import os
import pickle
import time
from pathlib import Path

import numpy as np

from shared.constants import EMOTION_LABELS, FEATURE_DIM
from shared.utils.logger import get_logger
from meta_learner import build_feature_vector
from trainer.utils import _run_parallel_batches, _vader

logger = get_logger("trainer")

_CSV_PATH     = Path(os.environ.get("TRAINING_DATA_DIR", "/app/training_data")) / "goemotions_cdm_10k.csv"
_MODEL_PATH   = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
_CSV_CACHE_ID = f"csv_local_v2_{FEATURE_DIM}"


def extract_csv_local_features(
    vader_analyzer, bert_analyzer, goe_analyzer,
    batch_size: int = 32,
) -> tuple:
    empty = np.empty((0, FEATURE_DIM), dtype=np.float32)

    if not _CSV_PATH.exists():
        logger.warning(f"[CSVLocal] CSV not found at {_CSV_PATH} — skipping.")
        return empty, []

    cache_path = _MODEL_PATH.parent / "csv_local_cache.pkl"
    csv_mtime  = str(_CSV_PATH.stat().st_mtime)
    cache_key  = f"{_CSV_CACHE_ID}_{csv_mtime}"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                X, y = cached["data"]
                logger.info(f"[CSVLocal] Cache hit — {len(y)} samples, skipping NLP.")
                return np.array(X, dtype=np.float32), y
            logger.info("[CSVLocal] Cache stale — recomputing.")
        except Exception as e:
            logger.warning(f"[CSVLocal] Cache load failed: {e} — recomputing.")

    from collections import Counter
    pairs: list = []
    label_counts: Counter = Counter()
    NEUTRAL_CAP = 500

    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("goemotions_label", "").strip()
            text  = row.get("text", "").strip()
            if not text or label not in EMOTION_LABELS:
                continue
            if label == "neutral" and label_counts["neutral"] >= NEUTRAL_CAP:
                continue
            pairs.append((text, label))
            label_counts[label] += 1

    if not pairs:
        logger.warning("[CSVLocal] No valid pairs loaded from CSV.")
        return empty, []

    logger.info(f"[CSVLocal] {len(pairs)} samples ({len(label_counts)} classes). Running NLP...")
    texts  = [t for t, _ in pairs]
    labels = [l for _, l in pairs]

    t0 = time.time()
    vader_outs, bert_outs, goe_outs = _run_parallel_batches(
        bert_analyzer, goe_analyzer, texts,
        vader_analyzer=vader_analyzer,
        batch_size=batch_size, label_prefix="CSVLocal",
    )
    logger.info(f"[CSVLocal] NLP done in {time.time()-t0:.1f}s")

    features = []
    for v_out, b_out, g_out in zip(vader_outs, bert_outs, goe_outs):
        fv = build_feature_vector({"vader": v_out, "basic_bert": b_out, "go_emotions": g_out})
        features.append(fv.flatten())

    X = np.array(features, dtype=np.float32)
    logger.info(f"[CSVLocal] Built {len(features)} feature vectors (dim={FEATURE_DIM}).")

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"cache_key": cache_key, "data": (X, labels)}, f)
        logger.info(f"[CSVLocal] Cache written to {cache_path}.")
    except Exception as e:
        logger.warning(f"[CSVLocal] Cache write failed: {e}")

    return X, labels
