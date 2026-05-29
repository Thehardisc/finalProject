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
from .analyzers   import _get_analyzers, _vader, _run
from .preprocessor import build_fv, filter_outliers, filter_balance
from .data_fetcher import fetch_live_data
from .reporter     import print_report

# Prometheus metrics — imported lazily so tests can stub if needed
try:
    import metrics as METRICS
except ImportError:
    METRICS = None

logger = get_logger("trainer")


def _phase_log(phase: str, started_at: float, **fields):
    """Emit a structured trainer_phase event with the elapsed duration."""
    duration_ms = round((time.time() - started_at) * 1000, 1)
    payload = {"event": "trainer_phase", "phase": phase, "duration_ms": duration_ms}
    payload.update(fields)
    logger.info(f"trainer_phase phase={phase} duration_ms={duration_ms}", extra={"extra_data": payload})


def run_one_cycle(reload_callback):
    """
    Full training + filter + gate + deploy cycle.
    Calls reload_callback(new_model) on success so main.py can hot-swap
    the global META_LEARNER without restarting the container.
    """
    cycle_started = time.time()
    logger.info(
        "trainer_cycle_start",
        extra={"extra_data": {
            "event":            "trainer_cycle_start",
            "max_samples":      MAX_SAMPLES,
            "accuracy_gate":    ACCURACY_GATE,
            "retrain_interval": RETRAIN_INTERVAL,
        }},
    )

    prev_meta = {}
    if META_PATH.exists():
        try:
            with open(META_PATH) as f:
                prev_meta = json.load(f)
        except Exception as e:
            logger.warning(
                "prev_meta_load_failed",
                extra={"extra_data": {"event": "prev_meta_load_failed", "error": str(e)}},
            )

    # ── Load GoEmotions dataset ───────────────────────────────────────────────
    t_phase = time.time()
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    except Exception as e:
        logger.error(
            "trainer_data_load_failed",
            extra={"extra_data": {"event": "trainer_data_load_failed", "error": str(e)}},
        )
        return
    _phase_log("data_load", t_phase, train_size=len(ds["train"]))
    t0 = time.time()  # kept for the per-sample timing logic below

    train_raw = list(ds["train"])[:MAX_SAMPLES]
    val_raw   = list(ds["validation"])[:MAX_SAMPLES // 5]
    test_raw  = list(ds["test"])[:MAX_SAMPLES // 5]

    # ── Load AI analyzers ─────────────────────────────────────────────────────
    import torch
    device = 0 if torch.cuda.is_available() else -1
    vader, bert, goe, emoji_scorer = _get_analyzers(device)

    rng = random.Random(42)

    def process(split, name):
        X, y, gs_list = [], [], []
        for i, s in enumerate(split):
            if i % 100 == 0:
                pct = (i / len(split)) * 100
                logger.info(f"  [Trainer] {name} Building Features: {i}/{len(split)} ({pct:.0f}%)")
            lids = s.get("labels", [])
            if not lids:
                continue
            valid_lids = [lid for lid in lids if lid < len(EMOTION_LABELS)]
            if not valid_lids:
                continue
            text = s["text"]
            vs = {f"vader_{k}": v for k, v in _vader(vader, text).items()}
            bs = _run(bert, text)
            gs = _run(goe,  text)
            es = emoji_scorer.analyze(text)
            gs_list.append(gs)

            # Layer 3: pick the label GoEmotions is most confident about
            # (not blindly labels[0]) — improves label quality for multi-label samples
            best_lid = max(valid_lids, key=lambda lid: gs.get(EMOTION_LABELS[lid], 0.0))
            synthetic_context = {
                "avg_valence":  rng.uniform(-1.0, 1.0),
                "prev_emotion": rng.choice(EMOTION_LABELS)
            }
            X.append(build_fv(vs, bs, gs, es, context=synthetic_context))
            y.append(EMOTION_LABELS[best_lid])
        pt = (time.time() - t0) / len(X) if X else 0
        logger.info(f"  [Trainer] {name}: {len(X)} samples. Avg {pt*1000:.2f}ms/sample.")
        return X, y, gs_list

    t_phase = time.time(); X_tr, y_tr, gs_tr = process(train_raw, "train"); _phase_log("feature_extract_train", t_phase, n=len(X_tr))
    t_phase = time.time(); X_v,  y_v,  _     = process(val_raw,   "val  ");  _phase_log("feature_extract_val",   t_phase, n=len(X_v))
    t_phase = time.time(); X_te, y_te, _     = process(test_raw,  "test ");  _phase_log("feature_extract_test",  t_phase, n=len(X_te))

    # ── Augment with live verified data ───────────────────────────────────────
    t_phase = time.time()
    X_live, y_live = fetch_live_data(vader, bert, goe, emoji_scorer)
    if X_live:
        weight = 3
        X_tr.extend(X_live * weight)
        y_tr.extend(y_live * weight)
        gs_tr.extend([{}] * len(X_live) * weight)
    _phase_log("live_aug", t_phase, n_live=len(X_live), live_weight=3 if X_live else 0)

    # ── Free analyzers from RAM ───────────────────────────────────────────────
    del vader, bert, goe, emoji_scorer
    gc.collect()

    # ── Filtering ─────────────────────────────────────────────────────────────
    t_phase = time.time()
    dist_before = Counter(y_tr).most_common(5)
    logger.log_stats("Pre-Filter Distribution (Top 5)", dict(dist_before))

    n_before = len(X_tr)
    X_tr, y_tr, n_out = filter_outliers(X_tr, y_tr, gs_tr)
    X_tr, y_tr        = filter_balance(X_tr, y_tr)
    n_filtered = n_before - len(X_tr)
    _phase_log("filter", t_phase, removed=n_filtered, outliers_removed=n_out, kept=len(X_tr))

    dist_after = Counter(y_tr).most_common(5)
    logger.log_stats("Post-Filter Distribution (Top 5)", dict(dist_after))

    if not X_tr:
        logger.error(
            "trainer_no_samples_after_filter",
            extra={"extra_data": {"event": "trainer_aborted", "reason": "no_samples_after_filter"}},
        )
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    t_phase = time.time()
    logger.info("⚡ PHASE: Ensemble (Voting) Pipeline Training...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(
                    max_iter=1000, C=1.0, solver='lbfgs',
                    multi_class='multinomial', class_weight='balanced',
                    random_state=42
                )),
                ('hgb', HistGradientBoostingClassifier(
                    max_iter=200, max_depth=6, learning_rate=0.05,
                    min_samples_leaf=10, random_state=42
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=150, class_weight='balanced',
                    max_depth=12, min_samples_leaf=5,
                    random_state=42, n_jobs=-1
                )),
            ],
            voting='soft',
            n_jobs=-1
        ))
    ])
    pipeline.fit(np.array(X_tr), y_tr)
    _phase_log("train", t_phase, n_train=len(X_tr))

    t_phase = time.time()
    y_v_pred  = pipeline.predict(np.array(X_v))
    y_te_pred = pipeline.predict(np.array(X_te))

    val_acc  = accuracy_score(y_v,  y_v_pred)
    test_acc = accuracy_score(y_te, y_te_pred)
    val_f1   = f1_score(y_v,  y_v_pred,  average='macro', zero_division=0)
    test_f1  = f1_score(y_te, y_te_pred, average='macro', zero_division=0)
    _phase_log("eval", t_phase,
               val_acc=round(val_acc, 4), test_acc=round(test_acc, 4),
               val_f1=round(val_f1, 4),   test_f1=round(test_f1, 4))

    # ── Accuracy gate + deploy ────────────────────────────────────────────────
    deployed = test_acc >= ACCURACY_GATE
    delta    = test_acc - ACCURACY_GATE
    logger.info(
        f"trainer_gate_decision test_acc={test_acc:.4f} threshold={ACCURACY_GATE} "
        f"delta={delta:+.4f} deployed={deployed}",
        extra={"extra_data": {
            "event":      "trainer_gate_decision",
            "test_acc":   round(test_acc, 4),
            "threshold":  ACCURACY_GATE,
            "delta":      round(delta, 4),
            "deployed":   deployed,
        }},
    )
    if METRICS is not None:
        METRICS.trainer_last_run_timestamp.set_to_current_time()
        METRICS.trainer_last_accuracy.set(float(test_acc))
        METRICS.trainer_runs_total.labels(result="accepted" if deployed else "rejected").inc()
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
        if METRICS is not None:
            METRICS.trainer_last_accepted_accuracy.set(float(test_acc))

        ready_marker = MODEL_PATH.parent / ".ready"
        ready_marker.touch()

        if not hasattr(start_trainer_thread, '_initial_trained'):
            logger.info("Training complete. Opening system gates.")
            setattr(start_trainer_thread, '_initial_trained', True)

    cycle_duration = time.time() - cycle_started
    print_report(prev_meta, test_acc, test_f1, len(X_tr), n_filtered, deployed,
                 duration_s=cycle_duration)
    logger.info(
        f"trainer_cycle_complete total_duration_s={cycle_duration:.1f} deployed={deployed}",
        extra={"extra_data": {
            "event":            "trainer_cycle_complete",
            "total_duration_s": round(cycle_duration, 1),
            "samples_trained":  len(X_tr),
            "test_accuracy":    round(test_acc, 4),
            "deployed":         deployed,
        }},
    )


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
                logger.error(
                    f"trainer_cycle_errored error={type(e).__name__}",
                    extra={"extra_data": {
                        "event":       "trainer_cycle_errored",
                        "error_class": type(e).__name__,
                        "error":       str(e),
                    }},
                )
                traceback.print_exc()
                if METRICS is not None:
                    METRICS.trainer_runs_total.labels(result="errored").inc()
            finally:
                if r:
                    r.set("system:training_in_progress", "0")

            # Sleep in 60s chunks emitting a heartbeat each tick — proves the loop
            # is alive between training cycles (which can be 30 min apart).
            remaining = RETRAIN_INTERVAL
            while remaining > 0:
                chunk = min(60, remaining)
                time.sleep(chunk)
                remaining -= chunk
                if r:
                    r.set("trainer:last_heartbeat", time.time(), ex=RETRAIN_INTERVAL * 3)
                logger.info(
                    f"trainer_heartbeat sleeping_for={remaining}s",
                    extra={"extra_data": {
                        "event":              "trainer_heartbeat",
                        "next_cycle_in_s":    remaining,
                        "retrain_interval_s": RETRAIN_INTERVAL,
                    }},
                )

    t = threading.Thread(target=_loop, name="trainer", daemon=True)
    t.start()
    return t
