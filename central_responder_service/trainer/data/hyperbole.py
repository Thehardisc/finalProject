import csv
import os
import pickle
import time
from pathlib import Path

import numpy as np

from shared.constants import EMOTION_LABELS, FEATURE_DIM
from shared.utils.logger import get_logger
from meta_learner import build_feature_vector
from trainer.utils import _run_parallel_batches

logger = get_logger("trainer")

_MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
_CSV_PATH   = _MODEL_PATH.parent / "hyperbole_samples.csv"
_CACHE_ID   = f"hyperbole_v1_{FEATURE_DIM}"
_MAX_ROWS   = int(os.environ.get("HYPERBOLE_MAX_SAMPLES", "500"))


# Loads Claude-generated hyperbole/slang samples (see hyperbole_gen.py) from .cache.
def extract_hyperbole_features(vader_analyzer, bert_analyzer, goe_analyzer, batch_size: int = 32) -> tuple:
    empty = np.empty((0, FEATURE_DIM), dtype=np.float32)

    if not _CSV_PATH.exists():
        logger.info(f"[Hyperbole] No sample file at {_CSV_PATH} — skipping (run hyperbole_gen.py to create).")
        return empty, []

    cache_path = _MODEL_PATH.parent / "hyperbole_cache.pkl"
    cache_key  = f"{_CACHE_ID}_{_CSV_PATH.stat().st_mtime}"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                X, y = cached["data"]
                logger.info(f"[Hyperbole] Cache hit — {len(y)} samples, skipping NLP.")
                return np.array(X, dtype=np.float32), y
        except Exception as e:
            logger.warning(f"[Hyperbole] Cache load failed: {e} — recomputing.")

    pairs = []
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("goemotions_label", "").strip()
            text  = row.get("text", "").strip()
            if text and label in EMOTION_LABELS:
                pairs.append((text, label))
            if len(pairs) >= _MAX_ROWS:
                break

    if not pairs:
        logger.warning("[Hyperbole] No valid pairs in sample file.")
        return empty, []

    logger.info(f"[Hyperbole] {len(pairs)} samples. Running NLP...")
    texts, labels = [t for t, _ in pairs], [l for _, l in pairs]
    t0 = time.time()
    vader_outs, bert_outs, goe_outs = _run_parallel_batches(
        bert_analyzer, goe_analyzer, texts,
        vader_analyzer=vader_analyzer, batch_size=batch_size, label_prefix="Hyperbole",
    )
    logger.info(f"[Hyperbole] NLP done in {time.time()-t0:.1f}s")

    features = [
        build_feature_vector({"vader": v, "basic_bert": b, "go_emotions": g}).flatten()
        for v, b, g in zip(vader_outs, bert_outs, goe_outs)
    ]
    X = np.array(features, dtype=np.float32)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"cache_key": cache_key, "data": (X, labels)}, f)
    except Exception as e:
        logger.warning(f"[Hyperbole] Cache write failed: {e}")

    return X, labels
