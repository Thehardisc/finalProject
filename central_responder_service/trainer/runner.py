"""
trainer/runner.py — Training cycle orchestration and background thread management.

run_one_cycle()       — full pipeline: load data → extract features → train → gate → deploy
start_trainer_thread() — spawns daemon thread that calls run_one_cycle() every RETRAIN_INTERVAL
"""
import os
import gc
import json
import time
import pickle
import random
import datetime
import threading
import traceback
import numpy as np
from pathlib import Path
from collections import Counter

import redis as redis_sync

from shared.utils.logger import get_logger
from shared.constants import EMOTION_LABELS

from .config import (RETRAIN_INTERVAL, ACCURACY_GATE, MAX_SAMPLES,
                     MODEL_PATH, META_PATH)
from .analyzers   import _get_analyzers, _vader, _run, _emojinet
from .preprocessor import build_fv, filter_outliers, filter_balance
from .data_fetcher import fetch_live_data
from .reporter     import print_report

logger = get_logger("trainer")


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
            with open(META_PATH) as f:
                prev_meta = json.load(f)
        except Exception:
            pass

    # ── Load GoEmotions dataset ───────────────────────────────────────────────
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

    # ── Load AI analyzers ─────────────────────────────────────────────────────
    import torch
    device = 0 if torch.cuda.is_available() else -1
    vader, bert, goe = _get_analyzers(device)

    rng = random.Random(42)

    def process(split, name):
        X, y, gs_list = [], [], []
        for i, s in enumerate(split):
            if i % 100 == 0:
                pct = (i / len(split)) * 100
                logger.info(f"  [Trainer] {name} Building Features: {i}/{len(split)} ({pct:.0f}%)")
            lids = s.get("labels", [])
            if not lids or lids[0] >= len(EMOTION_LABELS):
                continue
            text = s["text"]
            vs = {f"vader_{k}": v for k, v in _vader(vader, text).items()}
            bs = _run(bert, text)
            gs = _run(goe,  text)
            es = _emojinet(text)
            gs_list.append(gs)
            synthetic_context = {
                "avg_valence":  rng.uniform(-1.0, 1.0),
                "prev_emotion": rng.choice(EMOTION_LABELS)
            }
            X.append(build_fv(vs, bs, gs, es, context=synthetic_context))
            y.append(EMOTION_LABELS[lids[0]])
        pt = (time.time() - t0) / len(X) if X else 0
        logger.info(f"  [Trainer] {name}: {len(X)} samples. Avg {pt*1000:.2f}ms/sample.")
        return X, y, gs_list

    X_tr, y_tr, gs_tr = process(train_raw, "train")
    X_v,  y_v,  _     = process(val_raw,   "val  ")
    X_te, y_te, _     = process(test_raw,  "test ")

    # ── Augment with live verified data ───────────────────────────────────────
    X_live, y_live = fetch_live_data(vader, bert, goe)
    if X_live:
        weight = 3
        X_tr.extend(X_live * weight)
        y_tr.extend(y_live * weight)
        gs_tr.extend([{}] * len(X_live) * weight)
        logger.info(f"  [Trainer] Augmented with {len(X_live)} live verified samples (weight={weight})")

    # ── Free analyzers from RAM ───────────────────────────────────────────────
    del vader, bert, goe
    gc.collect()
    logger.info("Aggressively purged transient analyzers from RAM.")

    # ── Filtering ─────────────────────────────────────────────────────────────
    dist_before = Counter(y_tr).most_common(5)
    logger.log_stats("Pre-Filter Distribution (Top 5)", dict(dist_before))

    n_before = len(X_tr)
    logger.info("Applying bad-data filters...")
    X_tr, y_tr, n_out = filter_outliers(X_tr, y_tr, gs_tr)
    logger.info(f"  Layer 2 (outlier filter)  : removed {n_out}")
    X_tr, y_tr        = filter_balance(X_tr, y_tr)
    n_filtered = n_before - len(X_tr)

    dist_after = Counter(y_tr).most_common(5)
    logger.log_stats("Post-Filter Distribution (Top 5)", dict(dist_after))

    if not X_tr:
        logger.error("No samples remaining after filtering. Sequence aborted.")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("⚡ PHASE: Logistic Regression Pipeline Training...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                   multi_class='multinomial', class_weight='balanced',
                                   random_state=42))
    ])
    pipeline.fit(np.array(X_tr), y_tr)

    y_v_pred  = pipeline.predict(np.array(X_v))
    y_te_pred = pipeline.predict(np.array(X_te))

    val_acc  = accuracy_score(y_v,  y_v_pred)
    test_acc = accuracy_score(y_te, y_te_pred)
    val_f1   = f1_score(y_v,  y_v_pred,  average='macro', zero_division=0)
    test_f1  = f1_score(y_te, y_te_pred, average='macro', zero_division=0)
    logger.info(f"  [Metrics] Val  — Acc: {val_acc:.4f}  |  Macro-F1: {val_f1:.4f}")
    logger.info(f"  [Metrics] Test — Acc: {test_acc:.4f}  |  Macro-F1: {test_f1:.4f}")

    # ── Accuracy gate + deploy ────────────────────────────────────────────────
    deployed = test_acc >= ACCURACY_GATE
    if deployed:
        tmp = MODEL_PATH.with_suffix(".tmp.pkl")
        with open(tmp, 'wb') as f:
            pickle.dump(pipeline, f)

        # R4: back up the current model before overwriting so we can rollback
        if MODEL_PATH.exists():
            prev_path = MODEL_PATH.with_suffix(".prev.pkl")
            try:
                import shutil
                shutil.copy2(MODEL_PATH, prev_path)
                logger.info(f"Previous model backed up to '{prev_path.name}'.")
            except Exception as e:
                logger.warning(f"Could not back up previous model: {e}")

        tmp.rename(MODEL_PATH)

        # Write SHA-256 sidecar so loader.py can verify integrity next load
        import hashlib
        h = hashlib.sha256()
        with open(MODEL_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        with open(str(MODEL_PATH) + ".sha256", "w") as sf:
            sf.write(h.hexdigest())
        logger.info("SHA-256 sidecar updated.")

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

        reload_callback(pipeline)

        ready_marker = MODEL_PATH.parent / ".ready"
        ready_marker.touch()

        if not hasattr(start_trainer_thread, '_initial_trained'):
            logger.info("Training complete. Opening system gates.")
            setattr(start_trainer_thread, '_initial_trained', True)

    print_report(prev_meta, test_acc, test_f1, len(X_tr), n_filtered, deployed)


def start_trainer_thread(reload_callback):
    """
    Spawn a daemon thread that runs run_one_cycle() every RETRAIN_INTERVAL seconds.
    The reload_callback(new_model) is called after each successful deploy so
    main.py can swap the global META_LEARNER without restarting.
    """
    def _loop():
        logger.info("🚀 Trainer started.")
        logger.log_stats("Trainer Configuration (resolved from env)", {
            "RETRAIN_INTERVAL_SECONDS": RETRAIN_INTERVAL,
            "ACCURACY_GATE":            ACCURACY_GATE,
            "MAX_SAMPLES":              MAX_SAMPLES,
            "MODEL_PATH":               str(MODEL_PATH),
        })
        logger.info("API access will be enabled on first model completion.")

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
                    # R5: heartbeat so health checks can detect a crashed trainer
                    r.set("trainer:last_heartbeat", time.time(), ex=RETRAIN_INTERVAL * 3)
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
