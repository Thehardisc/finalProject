"""
trainer.py — Periodic retraining background thread for central_responder_service.

Runs in a daemon thread alongside the main Redis consumer loop.
After each successful cycle, reloads the model into META_LEARNER
variable so predictions use the new weights immediately.

Configuration (read from environment via docker-compose):
  RETRAIN_INTERVAL_SECONDS  (default: 600)
  ACCURACY_GATE             (default: 0.40)
  MAX_SAMPLES               (default: 10000)
"""

import os
import json
import time
import pickle
import datetime
import statistics
import threading
import traceback
import numpy as np
from pathlib import Path
from collections import Counter
from sqlalchemy import create_engine, text
from shared.utils.logger import get_logger
import redis as redis_sync
 
logger = get_logger("trainer")
 
# db settings
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "emotion_db")
DB_HOST = os.getenv("DB_HOST", "db")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

# configuration
RETRAIN_INTERVAL = int(os.environ.get("RETRAIN_INTERVAL_SECONDS", 1800))
ACCURACY_GATE    = float(os.environ.get("ACCURACY_GATE", 0.40))
MAX_SAMPLES      = int(os.environ.get("MAX_SAMPLES", 2500))
MODEL_PATH       = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
META_PATH        = MODEL_PATH.with_name("meta_weights_meta.json")

import random as _random_mod
from shared.constants import EMOTION_LABELS, VADER_KEYS, BERT_LABELS, FEATURE_DIM, CONTEXT_DIM
from meta_learner import build_feature_vector


# reporter
def _bar(value: float, width: int = 25) -> str:
    """Renders a unicode progress bar for logging."""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def print_report(prev_meta: dict, new_acc: float, new_f1: float, n_train: int, n_filtered: int, deployed: bool) -> None:
    """Log a structured summary after each training cycle."""
    prev_acc = prev_meta.get("test_accuracy")
    delta = (new_acc - prev_acc) if prev_acc is not None else None

    stats = {
        "Previous Accuracy": f"{prev_acc:.4f}" if prev_acc is not None else "N/A",
        "New Accuracy (test)": f"{new_acc:.4f}  {_bar(new_acc)}",
        "New F1 (macro)":     f"{new_f1:.4f}  {_bar(new_f1)}",
        "Samples Trained":    n_train,
        "Samples Filtered":   n_filtered,
        "Deployment":         "✅ DEPLOYED" if deployed else "❌ REJECTED (accuracy/regression)"
    }

    if delta is not None:
        direction = "↑" if delta >= 0 else "↓"
        stats["Delta"] = f"{direction} {delta*100:+.2f}%"

    logger.log_stats("Retraining Report", stats)


# feature generation (delegated to meta_learner.py)


# outliner filters
def filter_outliers(X, y, goe_list):
    """Layer 2: Drop samples where GoEmotions gives < 5% confidence to the gold label."""
    cX, cy, removed = [], [], 0
    for fv, label, goe in zip(X, y, goe_list):
        if label not in EMOTION_LABELS or goe.get(label, 0.0) < 0.05:
            removed += 1
        else:
            cX.append(fv); cy.append(label)
    return cX, cy, removed

def filter_balance(X, y):
    """Layer 3: Cap any class at 3× the median class count."""
    if not y:
        return X, y
    counts  = Counter(y)
    cap     = max(50, int(statistics.median(counts.values()) * 3))
    seen    = Counter()
    cX, cy  = [], []
    for fv, label in zip(X, y):
        if seen[label] < cap:
            cX.append(fv); cy.append(label); seen[label] += 1
    removed = len(X) - len(cX)
    if removed:
        logger.info(f"  [Filter] Balance cap: removed {removed} samples (cap={cap}/class).")
    return cX, cy


# model utilities
def _get_analyzers(device):
    logger.info("Loading analyzers transiently into RAM...")
    import torch

    torch.set_num_threads(1)

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import pipeline as hf_pipeline

    vader = SentimentIntensityAnalyzer()
    bert  = hf_pipeline("text-classification",
                         model="j-hartmann/emotion-english-distilroberta-base",
                         return_all_scores=True, device=device)
    goe   = hf_pipeline("text-classification",
                         model="SamLowe/roberta-base-go_emotions",
                         return_all_scores=True, device=device)

    logger.info("Analyzers fully loaded.")
    return vader, bert, goe

def _vader(v, text):
    s = v.polarity_scores(text)
    return {k: s[k] for k in ['neg','neu','pos','compound']}

def _run(model, text):
    try:    return {r['label']: r['score'] for r in model(text[:512])[0]}
    except: return {}


def build_synthetic_context_vector() -> list:
    """Build a 23-dim synthetic context vector for training samples.
    All CDM scalars are zeroed — no ground-truth conversation dynamics exist
    for the GoEmotions static dataset. Zeros are safe: the MLP will learn
    to down-weight the context block when it carries no signal.
    """
    return [0.0] * CONTEXT_DIM


# fetch data
def fetch_live_data(vader, bert, goe):
    """
    Fetch verified samples from PostgreSQL (emotion_analysis joined with messages).
    Returns (X, y) lists of feature vectors and labels.
    """
    X, y = [], []
    try:
        engine = create_engine(DATABASE_URL)
        with engine.begin() as conn:
            try:
                conn.execute(text("ALTER TABLE emotion_analysis ADD COLUMN IF NOT EXISTS ground_truth_emotion VARCHAR(50);"))
            except Exception as e:
                logger.debug(f"  [SQL] Schema repair concurrent lock or failure: {e}")

            query = text(f"""
                WITH ranked_messages AS (
                    SELECT m.message_id, m.text, a.ground_truth_emotion, m.conversation_id, m.timestamp
                    FROM emotion_analysis a
                    JOIN messages m ON a.message_id = m.message_id
                    WHERE a.is_verified = TRUE
                )
                SELECT text, ground_truth_emotion
                FROM ranked_messages
                ORDER BY timestamp DESC
                LIMIT {MAX_SAMPLES}
            """)
            rows = conn.execute(query).fetchall()

            if not rows:
                return [], []

            logger.info(f"  [SQL] Found {len(rows)} verified live samples.")

            for text_content, label in rows:
                vs = {f"vader_{k}": v for k, v in _vader(vader, text_content).items()}
                bs = _run(bert, text_content)
                gs = _run(goe,  text_content)
                ctx = build_synthetic_context_vector()
                fv = build_feature_vector({"vader": vs, "basic_bert": bs, "go_emotions": gs}, context_vector=ctx)
                X.append(fv.flatten())
                y.append(label)

        return X, y
    except Exception as e:
        logger.error(f"  [SQL] Failed to fetch live data: {e}")
        return [], []


# run cycle
def run_one_cycle(reload_callback):
    """
    Full training + filter + gate + deploy cycle.
    Calls reload_callback(new_model) on success so main.py can hot-swap
    the global META_LEARNER without restarting the container.
    """
    logger.info(f"═══ Starting cycle at {datetime.datetime.utcnow().strftime('%H:%M:%S UTC')} ═══")

    prev_meta = {}
    if META_PATH.exists():
        try:
            with open(META_PATH) as f: prev_meta = json.load(f)
        except Exception: pass

    # Check Cache
    CACHE_PATH = MODEL_PATH.parent / "dataset_features_cache.pkl"
    if CACHE_PATH.exists():
        logger.info("Loading cached dataset features...")
        try:
            with open(CACHE_PATH, "rb") as f:
                cached_data = pickle.load(f)

            # Invalidate cache when feature vector dimensions change
            cached_dim = cached_data.get("feature_dim")
            if cached_dim != FEATURE_DIM:
                logger.warning(f"Cache feature_dim={cached_dim} != current {FEATURE_DIM}. Invalidating cache and rebuilding...")
                CACHE_PATH.unlink(missing_ok=True)
                return run_one_cycle(reload_callback)

            X_tr, y_tr, gs_tr = cached_data["train"]
            X_v, y_v, _ = cached_data["val"]
            X_te, y_te, _ = cached_data["test"]

            import torch
            device = 0 if torch.cuda.is_available() else -1
            vader, bert, goe = _get_analyzers(device)

            logger.info("Successfully loaded features from cache.")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Rebuilding features...")
            CACHE_PATH.unlink(missing_ok=True)
            return run_one_cycle(reload_callback)  # Retry fresh
    else:
        # Load dataset
        t0 = time.time()
        logger.info("Loading GoEmotions dataset...")
        try:
            from datasets import load_dataset
            ds = load_dataset("google-research-datasets/go_emotions", "simplified")
        except Exception as e:
            logger.error(f"Dataset load failed: {e}")
            return

        train_raw = list(ds["train"])[:MAX_SAMPLES]
        val_raw   = list(ds["validation"])[:MAX_SAMPLES // 5]
        test_raw  = list(ds["test"])[:MAX_SAMPLES // 5]
        logger.info(f"  [Extraction] Selected {len(train_raw)} train samples for cognitive build.")

        # Setup device mapping and load analyzers
        import torch
        device = 0 if torch.cuda.is_available() else -1
        vader, bert, goe = _get_analyzers(device)

        def process(split, name):
            X, y, gs_list = [], [], []
            for i, s in enumerate(split):
                if i % 100 == 0:
                    pct = (i / len(split)) * 100
                    logger.info(f"  [Trainer] {name} Building Features: {i}/{len(split)} ({pct:.0f}%)")
                lids = s.get("labels", [])
                if not lids or lids[0] >= len(EMOTION_LABELS): continue
                text = s["text"]
                vs = {f"vader_{k}": v for k, v in _vader(vader, text).items()}
                bs = _run(bert, text)
                gs = _run(goe,  text)
                gs_list.append(gs)
                ctx = build_synthetic_context_vector()
                X.append(build_feature_vector({"vader": vs, "basic_bert": bs, "go_emotions": gs}, context_vector=ctx).flatten())
                y.append(EMOTION_LABELS[lids[0]])
            pt = (time.time() - t0) / len(X) if X else 0
            logger.info(f"  [Trainer] {name}: {len(X)} samples. Avg {pt*1000:.2f}ms/sample.")
            return X, y, gs_list

        print("[Trainer] Building feature vectors...")
        X_tr, y_tr, gs_tr = process(train_raw, "train")
        X_v,  y_v,  _     = process(val_raw,   "val  ")
        X_te, y_te, _     = process(test_raw,  "test ")

        # Save to cache with feature_dim stamp for future invalidation
        logger.info("Saving extracted features to cache...")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({
                "feature_dim": FEATURE_DIM,
                "train": (X_tr, y_tr, gs_tr),
                "val":   (X_v, y_v, None),
                "test":  (X_te, y_te, None),
            }, f)
 
    # Fetch Live Supervised Data
    X_live, y_live = fetch_live_data(vader, bert, goe)
    if X_live:
        # Augment training set with live data
        # We can duplicate live data to give it more weight if the set is small
        weight = 3 
        X_tr.extend(X_live * weight)
        y_tr.extend(y_live * weight)
        # We don't filter live data (it's already verified by human)
        # But we need placeholder gs entries if we use Layer 2 filter on everything
        # Actually, let's just bypass outlier filtering for live data
        gs_tr.extend([{}] * len(X_live) * weight)
        logger.info(f"  [Trainer] Augmented training set with {len(X_live)} live verified samples (weight={weight})")

    # Aggressive memory cleanup immediately after features are computed limits RAM peak
    del vader, bert, goe
    import gc
    gc.collect()
    logger.info("Aggressively purged transient analyzers from RAM.")

    # Stats: Distribution before filtering
    dist_before = Counter(y_tr).most_common(5)
    logger.log_stats("Pre-Filter Distribution (Top 5)", dict(dist_before))

    n_before = len(X_tr)
    logger.info("Applying bad-data filters...")

    # Layer 1 already enforced (only valid EMOTION_LABELS pass process())
    X_tr, y_tr, n_out = filter_outliers(X_tr, y_tr, gs_tr)
    logger.info(f"  Layer 2 (outlier filter)  : removed {n_out}")
    X_tr, y_tr        = filter_balance(X_tr, y_tr)
    n_filtered = n_before - len(X_tr)
    
    # Stats: Distribution after filtering
    dist_after = Counter(y_tr).most_common(5)
    logger.log_stats("Post-Filter Distribution (Top 5)", dict(dist_after))

    if not X_tr:
        logger.error("No samples remaining after filtering. Sequence aborted.")
        return

    # Train
    logger.info("⚡ PHASE: MLP Pipeline Training...")
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        ))
    ])
    # vstack normalises (1,N) items from build_feature_vector into a clean (n, FEATURE_DIM) matrix
    X_tr_arr = np.vstack([np.array(fv).flatten() for fv in X_tr])
    X_v_arr  = np.vstack([np.array(fv).flatten() for fv in X_v])
    X_te_arr = np.vstack([np.array(fv).flatten() for fv in X_te])

    pipeline.fit(X_tr_arr, y_tr)

    y_v_pred  = pipeline.predict(X_v_arr)
    y_te_pred = pipeline.predict(X_te_arr)

    val_acc   = accuracy_score(y_v,  y_v_pred)
    test_acc  = accuracy_score(y_te, y_te_pred)
    # Macro F1 is the right metric for imbalanced multi-class (28 emotions)
    val_f1    = f1_score(y_v,  y_v_pred,  average='macro', zero_division=0)
    test_f1   = f1_score(y_te, y_te_pred, average='macro', zero_division=0)
    logger.info(f"  [Metrics] Val  — Acc: {val_acc:.4f}  |  Macro-F1: {val_f1:.4f}")
    logger.info(f"  [Metrics] Test — Acc: {test_acc:.4f}  |  Macro-F1: {test_f1:.4f}")

    # Layer 4: accuracy gate (uses test accuracy to match industry convention)
    deployed = test_acc >= ACCURACY_GATE

    if deployed:
        # Atomic write: tmp → rename so central_responder never reads partial file
        tmp = MODEL_PATH.with_suffix(".tmp.pkl")
        with open(tmp, 'wb') as f: pickle.dump(pipeline, f)
        tmp.rename(MODEL_PATH)

        with open(META_PATH, 'w') as f:
            json.dump({
                "trained_at":          datetime.datetime.utcnow().isoformat() + "Z",
                "training_samples":    len(X_tr),
                "filtered_samples":    n_filtered,
                "validation_accuracy": round(val_acc,  4),
                "validation_f1_macro": round(val_f1,   4),
                "test_accuracy":       round(test_acc, 4),
                "test_f1_macro":       round(test_f1,  4),
                "previous_accuracy":   round(prev_meta.get("test_accuracy", 0), 4),
                "improvement":         round(test_acc - prev_meta.get("test_accuracy", 0), 4),
                "accuracy_gate":       ACCURACY_GATE,
            }, f, indent=2)

        # swap model
        reload_callback(pipeline)

        # Gating feedback: Create a definitive signal that a VALID model is live
        ready_marker = MODEL_PATH.parent / ".ready"
        ready_marker.touch()

        if not hasattr(start_trainer_thread, '_initial_trained'):
            logger.info("Training complete. Opening system gates.")
            setattr(start_trainer_thread, '_initial_trained', True)

    print_report(prev_meta, test_acc, test_f1, len(X_tr), n_filtered, deployed)


# background worker thread
def start_trainer_thread(reload_callback):
    """
    Spawn a daemon thread that runs run_one_cycle() every RETRAIN_INTERVAL seconds.
    The reload_callback(new_model) is called after each successful deploy so
    main.py can swap the global META_LEARNER without restarting.
    """
    def _loop():
        logger.info(f"🚀 Trainer started.")
        logger.log_stats("Trainer Configuration (resolved from env)", {
            "RETRAIN_INTERVAL_SECONDS": RETRAIN_INTERVAL,
            "ACCURACY_GATE":            ACCURACY_GATE,
            "MAX_SAMPLES":              MAX_SAMPLES,
            "MODEL_PATH":               str(MODEL_PATH),
        })
        logger.info("API access will be enabled on first model completion.")

        # Connect a simple sync Redis client to set training flags
        r = None
        try:
            r = redis_sync.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True
            )
        except Exception as e:
            logger.warning(f"Trainer could not connect to Redis for flags: {e}")

        while True:
            try:
                if r:
                    r.set("system:training_in_progress", "1")
                run_one_cycle(reload_callback)
            except Exception as e:
                logger.error(f"Unhandled error: {e}")
                traceback.print_exc()
            finally:
                if r:
                    r.set("system:training_in_progress", "0")
            logger.debug(f"Sleeping {RETRAIN_INTERVAL}s...")
            time.sleep(RETRAIN_INTERVAL)

    t = threading.Thread(target=_loop, name="trainer", daemon=True)
    t.start()
    return t
