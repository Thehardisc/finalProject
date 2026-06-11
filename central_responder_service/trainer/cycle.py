"""
trainer/cycle.py — Main training orchestrator: run_one_cycle() and start_trainer_thread().
"""

import datetime
import gc
import json
import os
import pickle
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import redis as redis_sync
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

from shared.constants import EMOTION_LABELS, FEATURE_DIM
from shared.utils.logger import get_logger
from trainer.utils import (
    _get_analyzers, filter_outliers, filter_balance, print_report, _bar,
)
from trainer.data.synthetic import (
    build_synthetic_context_vector, load_synthetic_features,
    _SYNTHETIC_CLASSES,
)
from trainer.data.empathetic import extract_empathetic_dialogues_features
from trainer.data.goemotions import extract_goemotions_direct_features
from trainer.data.meld import extract_meld_features, _download_meld_raw
from trainer.data.db import fetch_live_data, load_relabeled_conversations
from trainer.models import train_gating_network

logger = get_logger("trainer")

# ── Configuration ───────────────────────────────────────────────────────────────
RETRAIN_INTERVAL        = int(os.environ.get("RETRAIN_INTERVAL_SECONDS",   1800))
ACCURACY_GATE           = float(os.environ.get("ACCURACY_GATE",            0.40))
MAX_EMPATHETIC_SAMPLES  = int(os.environ.get("MAX_EMPATHETIC_SAMPLES",     25_000))
MIN_DB_SAMPLES          = int(os.environ.get("MIN_DB_SAMPLES",             50))
MODEL_PATH              = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
META_PATH               = MODEL_PATH.with_name("meta_weights_meta.json")

RELOAD_CHANNEL = "model_reload_signal"


def run_one_cycle(reload_callback=None) -> None:
    """
    Full build → filter → train → gate → deploy cycle.

    Two phases:
      Bootstrap  — runs once when no model file exists.
                   Trains on EmpatheticDialogues (size: MAX_EMPATHETIC_SAMPLES).
                   Results are cached so NLP inference isn't repeated every cycle.
      Continuous — every subsequent cycle trains only on verified PostgreSQL data
                   + relabeled conversations. Skips if fewer than MIN_DB_SAMPLES
                   samples are available.

    On success, calls reload_callback(wrapper) so main.py hot-swaps the
    global META_LEARNER without restarting the container.
    """
    cycle_start = time.time()
    logger.info(f"═══ Starting cycle at {datetime.datetime.utcnow():%H:%M:%S UTC} ═══")
    phase_times = {}  # track wall-clock seconds per phase

    prev_meta: dict = {}
    if META_PATH.exists():
        try:
            with open(META_PATH) as f:
                prev_meta = json.load(f)
        except Exception:
            pass

    is_bootstrap = not MODEL_PATH.exists()

    cpu_count = os.cpu_count() or 4

    if torch.cuda.is_available():
        device = 0
        nlp_batch_size = 128
        logger.info("Training device: CUDA GPU — NLP batch_size=128")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        # Apple Unified Memory silently swaps to SSD if we exceed physical RAM.
        # Batch size 1024 uses >12GB of RAM for attention matrices alone, causing swapping!
        # We cap it at 128 to stay entirely within fast physical RAM.
        nlp_batch_size = 128
        logger.info(f"Training device: Apple MPS ({cpu_count} CPU cores) — NLP batch_size=128")



    else:
        device = -1
        nlp_batch_size = 64
        logger.info(f"Training device: CPU ({cpu_count} cores) — NLP batch_size=64")

    # Maximize CPU utilization for NLP inference.
    # BERT and GoEmotions run in parallel threads; PyTorch releases the GIL during
    # C++ ops, so each model can use its own thread pool. Setting num_threads high
    # lets each model saturate the CPU during its forward pass.
    try:
        torch.set_num_threads(max(4, cpu_count - 1))
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(2)  # 2 models run in parallel
    except RuntimeError:
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logger.info(
        f"  PyTorch: {torch.get_num_threads()} intra-op threads, "
        f"{torch.get_num_interop_threads()} inter-op threads, "
        f"tokenizer parallelism=ON"
    )

    t_analyzers = time.time()
    vader, bert, goe = _get_analyzers(device)
    phase_times["Analyzer loading"] = time.time() - t_analyzers
    logger.info(f"  ⏱ Analyzers loaded in {phase_times['Analyzer loading']:.1f}s")

    if is_bootstrap:
        # ── One-time bootstrap from EmpatheticDialogues ────────────────────────
        CACHE_PATH = MODEL_PATH.parent / "dataset_features_cache.pkl"
        DATASET_ID = "empathetic_dialogues_multiturn_v1"

        if CACHE_PATH.exists():
            logger.info("Loading cached bootstrap features...")
            try:
                with open(CACHE_PATH, "rb") as f:
                    cached = pickle.load(f)
                if (cached.get("dataset_id") != DATASET_ID or
                        cached.get("feature_dim") != FEATURE_DIM or
                        cached.get("n_train_cap") != MAX_EMPATHETIC_SAMPLES):
                    logger.warning("Cache stale (dataset_id or feature_dim mismatch). Rebuilding.")
                    CACHE_PATH.unlink(missing_ok=True)
                    del vader, bert, goe
                    gc.collect()
                    return run_one_cycle(reload_callback)
                X_tr, y_tr, has_cdm_emp_tr = cached["train"]
                X_v,  y_v,  _              = cached["val"]
                X_te, y_te, _              = cached["test"]
                logger.info(
                    f"Bootstrap cache loaded — "
                    f"{len(X_tr)} train / {len(X_v)} val / {len(X_te)} test samples."
                )
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Rebuilding.")
                CACHE_PATH.unlink(missing_ok=True)
                try:
                    del vader, bert, goe
                except UnboundLocalError:
                    pass
                gc.collect()
                return run_one_cycle(reload_callback)
        else:
            logger.info(
                f"No model found — bootstrapping from EmpatheticDialogues "
                f"(MAX_EMPATHETIC_SAMPLES={MAX_EMPATHETIC_SAMPLES})..."
            )
            X_tr, y_tr, has_cdm_emp_tr = extract_empathetic_dialogues_features(vader, bert, goe, split="train", batch_size=nlp_batch_size)
            X_v,  y_v,  _              = extract_empathetic_dialogues_features(vader, bert, goe, split="val",   batch_size=nlp_batch_size)
            X_te, y_te, _              = extract_empathetic_dialogues_features(vader, bert, goe, split="test",  batch_size=nlp_batch_size)

            if not X_tr.size:
                logger.error("EmpatheticDialogues extraction returned empty train set. Aborting.")
                del vader, bert, goe
                gc.collect()
                return

            logger.info(
                f"Bootstrap: {len(X_tr)} train / {len(X_v)} val / {len(X_te)} test samples. "
                f"Saving feature cache..."
            )
            with open(CACHE_PATH, "wb") as f:
                pickle.dump({
                    "dataset_id":  DATASET_ID,
                    "feature_dim": FEATURE_DIM,
                    "n_train_cap": MAX_EMPATHETIC_SAMPLES,
                    "train": (list(X_tr), y_tr, list(has_cdm_emp_tr)),
                    "val":   (list(X_v),  y_v,  None),
                    "test":  (list(X_te), y_te, None),
                }, f)

            X_tr, X_v, X_te = list(X_tr), list(X_v), list(X_te)

        has_cdm_tr: list = list(has_cdm_emp_tr)
        gs_tr = [{label: 1.0} for label in y_tr]
        n_ed = len(X_tr)

        # ── Synthetic augmentation for the 5 missing GoEmotions classes ────────
        X_syn, y_syn, gs_syn = load_synthetic_features(vader, bert, goe, batch_size=nlp_batch_size)
        n_syn = 0
        if len(X_syn) > 0:
            n_syn      = len(X_syn)
            X_tr       = list(X_tr) + list(X_syn)
            y_tr       = y_tr + y_syn
            gs_tr      = gs_tr + gs_syn
            has_cdm_tr = has_cdm_tr + [False] * n_syn
            logger.info(
                f"Bootstrap: +{n_syn} synthetic samples "
                f"({_SYNTHETIC_CLASSES}) → {len(X_tr)} total train"
            )

        # ── GoEmotions direct (NLP-aligned) augmentation ─────────────────────
        from sklearn.model_selection import train_test_split as _tts
        X_goe_d, y_goe_d, gs_goe_d = extract_goemotions_direct_features(vader, bert, goe, batch_size=nlp_batch_size)
        n_goe_d = 0
        n_goe_val = 0
        n_goe_test = 0
        if len(X_goe_d) > 0:
            X_goe_arr = np.array(X_goe_d, dtype=np.float32)
            X_g_tr, X_g_hold, y_g_tr, y_g_hold = _tts(
                X_goe_arr, y_goe_d, test_size=0.30, random_state=42, stratify=y_goe_d
            )
            X_g_val, X_g_te, y_g_val, y_g_te = _tts(
                X_g_hold, y_g_hold, test_size=0.50, random_state=42, stratify=y_g_hold
            )

            from collections import Counter as _Counter
            c_tr  = _Counter(y_g_tr)
            c_val = _Counter(y_g_val)
            c_te  = _Counter(y_g_te)
            _w = 62
            logger.info(f"+{'-' * (_w - 2)}+")
            logger.info(f"|{'  GoEmotions-Direct Split (70 / 15 / 15) ':─^{_w - 2}}|")
            logger.info(f"+{'-' * (_w - 2)}+")
            logger.info(f"|  {'Emotion':<18}  {'Train':>6}  {'×3':>6}  {'Val':>5}  {'Test':>5}  |")
            logger.info(f"+{'-' * (_w - 2)}+")
            for lbl in EMOTION_LABELS:
                tr_n  = c_tr.get(lbl, 0)
                val_n = c_val.get(lbl, 0)
                te_n  = c_te.get(lbl, 0)
                tag   = "  " if (tr_n > 0 and val_n > 0 and te_n > 0) else "⚠ "
                logger.info(f"|  {tag}{lbl:<18}  {tr_n:>6}  {tr_n*3:>6}  {val_n:>5}  {te_n:>5}  |")
            logger.info(f"+{'-' * (_w - 2)}+")
            logger.info(f"|  {'TOTAL':<18}  {len(y_g_tr):>6}  {len(y_g_tr)*3:>6}  {len(y_g_val):>5}  {len(y_g_te):>5}  |")
            logger.info(f"+{'-' * (_w - 2)}+")

            n_goe_d    = len(X_g_tr) * 3
            X_tr       = list(X_tr) + list(X_g_tr) * 3
            y_tr       = y_tr + list(y_g_tr) * 3
            gs_tr      = gs_tr + gs_goe_d[:len(X_g_tr)] * 3
            has_cdm_tr = has_cdm_tr + [False] * n_goe_d

            n_goe_val  = len(X_g_val)
            n_goe_test = len(X_g_te)
            X_v  = list(X_v)  + list(X_g_val)
            y_v  = list(y_v)  + list(y_g_val)
            X_te = list(X_te) + list(X_g_te)
            y_te = list(y_te) + list(y_g_te)

            logger.info(
                f"Bootstrap: GoEmotions-direct split → "
                f"train +{n_goe_d} (3×) / val +{n_goe_val} / test +{n_goe_test} "
                f"→ {len(X_tr)} total train"
            )

        # ── MELD (real multi-turn context) augmentation ───────────────────────
        X_meld, y_meld, has_cdm_meld = extract_meld_features(vader, bert, goe, batch_size=nlp_batch_size)
        n_meld = 0
        n_meld_ctx = 0
        if len(X_meld) > 0:
            n_meld     = len(X_meld)
            n_meld_ctx = sum(has_cdm_meld)
            X_tr       = list(X_tr) + list(X_meld)
            y_tr       = y_tr + y_meld
            gs_tr      = gs_tr + [{label: 1.0} for label in y_meld]
            has_cdm_tr = has_cdm_tr + list(has_cdm_meld)
            logger.info(
                f"Bootstrap: +{n_meld} MELD samples "
                f"({n_meld_ctx} with real context) → {len(X_tr)} total train"
            )

        dataset_composition = {
            "EmpatheticDialogues (train)": n_ed,
            "Synthetic (train, 5 classes)": n_syn,
            f"GoEmotions-direct (train 3×)": n_goe_d,
            f"GoEmotions-direct (val)": n_goe_val,
            f"GoEmotions-direct (test)": n_goe_test,
            f"MELD ({n_meld_ctx} w/ real ctx)": n_meld,
        }

    else:
        # ── Continuous learning from database + cached external datasets ────────
        from sklearn.model_selection import train_test_split as _tts

        X_live, y_live = fetch_live_data(vader, bert, goe)
        X_rel,  y_rel  = load_relabeled_conversations()

        X_db = X_live * 3 + list(X_rel) * 3
        y_db = y_live * 3 + list(y_rel) * 3

        if len(X_db) < MIN_DB_SAMPLES:
            logger.info(
                f"Only {len(X_db)} DB samples (need ≥ MIN_DB_SAMPLES={MIN_DB_SAMPLES}). "
                f"Skipping cycle — waiting for more verified data."
            )
            del vader, bert, goe
            gc.collect()
            return

        logger.info(f"  [DB] {len(X_db)} samples available for continuous learning.")

        # GoEmotions direct (balanced 500/class × 28 classes) — loads from disk cache
        # when available, no NLP inference needed. Dramatically expands training set.
        X_goe_d, y_goe_d, _ = extract_goemotions_direct_features(
            vader, bert, goe, batch_size=nlp_batch_size
        )
        n_goe_cont = len(X_goe_d)
        if n_goe_cont > 0:
            logger.info(f"  [GoEDirect] +{n_goe_cont} cached GoEmotions samples added to continuous cycle.")

        # MELD (multi-turn with real CDM context) — loads from cache if available,
        # otherwise runs NLP inference on meld_raw_cache.json.
        X_meld_c, y_meld_c, has_cdm_meld_c = extract_meld_features(
            vader, bert, goe, batch_size=nlp_batch_size
        )
        n_meld_cont = len(X_meld_c)
        if n_meld_cont > 0:
            logger.info(f"  [MELD] +{n_meld_cont} MELD samples added to continuous cycle.")

        X_all = X_db + list(X_goe_d) + list(X_meld_c)
        y_all = y_db + list(y_goe_d) + list(y_meld_c)
        has_cdm_all = [False] * len(X_db) + [False] * n_goe_cont + list(has_cdm_meld_c)

        X_tr, X_v, y_tr, y_v, hc_tr, _ = _tts(
            X_all, y_all, has_cdm_all,
            test_size=0.20, random_state=42, stratify=y_all,
        )
        X_te, y_te = X_v, y_v
        gs_tr      = [{label: 1.0} for label in y_tr]
        has_cdm_tr = hc_tr
        dataset_composition = {
            "Live DB samples (3×)":           len(X_live) * 3,
            "Relabeled conversations (3×)":   len(X_rel)  * 3,
            "GoEmotions-direct (cache)":      n_goe_cont,
            f"MELD (cache, {sum(has_cdm_meld_c)} w/ real ctx)": n_meld_cont,
        }
        logger.info(
            f"Continuous cycle: {len(X_tr)} train / {len(X_v)} val samples "
            f"({len(X_db)} DB + {n_goe_cont} GoE + {n_meld_cont} MELD)."
        )

    del vader, bert, goe
    gc.collect()
    phase_times["Data loading (total)"] = time.time() - cycle_start
    logger.info(
        f"Transient analysers purged. "
        f"⏱ Data phase complete in {phase_times['Data loading (total)']:.1f}s"
    )

    # ── Dataset composition summary ────────────────────────────────────────────
    comp_rows = {src: cnt for src, cnt in dataset_composition.items() if cnt > 0}
    comp_rows["─── Totals ───"] = ""
    comp_rows["  Train samples"] = len(X_tr)
    comp_rows["  Val samples  "] = len(X_v)
    comp_rows["  Test samples "] = len(X_te)
    logger.log_stats("Dataset Composition (before training)", comp_rows)

    # ── Filters ────────────────────────────────────────────────────────────────
    t_filter = time.time()
    dist_before = Counter(y_tr).most_common(5)
    logger.log_stats("Pre-Filter Distribution (Top 5)", dict(dist_before))

    n_before = len(X_tr)
    if is_bootstrap:
        logger.info("  Layer 2 (outlier): skipped for bootstrap (curated dataset)")
        n_out = 0
    else:
        X_tr, y_tr, n_out, has_cdm_tr = filter_outliers(X_tr, y_tr, gs_tr, has_cdm_tr)
        logger.info(f"  Layer 2 (outlier): removed {n_out}")
    X_tr, y_tr, has_cdm_tr        = filter_balance(X_tr, y_tr, has_cdm_tr)
    n_filtered                    = n_before - len(X_tr)

    n_real_cdm = sum(has_cdm_tr)
    logger.info(
        f"  [CDMMask] {n_real_cdm}/{len(X_tr)} samples have real CDM "
        f"({100*n_real_cdm/max(len(X_tr),1):.1f}%)"
    )

    logger.log_stats("Post-Filter Distribution (Top 5)", dict(Counter(y_tr).most_common(5)))
    phase_times["Filtering"] = time.time() - t_filter
    logger.info(f"  ⏱ Filtering complete in {phase_times['Filtering']:.1f}s")

    if not X_tr:
        logger.error("No samples after filtering. Aborting.")
        return

    # ── Build numpy arrays ─────────────────────────────────────────────────────
    X_tr_arr = np.vstack([np.array(fv).flatten() for fv in X_tr])
    X_v_arr  = np.vstack([np.array(fv).flatten() for fv in X_v])
    X_te_arr = np.vstack([np.array(fv).flatten() for fv in X_te])

    # ── Train GatingEnsembleNet ────────────────────────────────────────────────
    t_train = time.time()
    logger.info("⚡ PHASE: GatingEnsembleNet Training (v2)...")
    if torch.cuda.is_available():
        train_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        train_device = "mps"
    else:
        train_device = "cpu"

    # Scale epochs and batch size to dataset size: larger datasets train better
    # with bigger batches (less noisy gradients) and more epochs.
    n_samples = len(X_tr_arr)
    dynamic_epochs    = 120 if n_samples > 5000 else 80
    dynamic_batch     = 512 if n_samples > 5000 else 256
    dynamic_patience  = 20  if n_samples > 5000 else 10
    dynamic_lb_coeff  = 1e-3 if n_samples > 5000 else 5e-4  # stronger balance on large sets
    logger.info(
        f"  [HParams] n={n_samples}: epochs={dynamic_epochs}  batch={dynamic_batch}  "
        f"patience={dynamic_patience}  lb_coeff={dynamic_lb_coeff}"
    )

    wrapper = train_gating_network(
        X_tr=X_tr_arr,
        y_tr=y_tr,
        X_v=X_v_arr,
        y_v=y_v,
        classes=EMOTION_LABELS,
        has_cdm=np.array(has_cdm_tr, dtype=bool),
        n_epochs=dynamic_epochs,
        batch_size=dynamic_batch,
        lr=5e-4,
        weight_decay=1e-4,
        load_balance_coeff=dynamic_lb_coeff,
        patience=dynamic_patience,
        device=train_device,
    )

    phase_times["Model training"] = time.time() - t_train
    _train_secs = phase_times['Model training']
    if _train_secs >= 60:
        _train_str = f"{int(_train_secs // 60)}m {int(_train_secs % 60)}s"
    else:
        _train_str = f"{_train_secs:.1f}s"
    logger.info(f"  ⏱ Model training phase complete in {_train_str}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    t_eval = time.time()
    y_v_pred  = wrapper.predict(X_v_arr)
    y_te_pred = wrapper.predict(X_te_arr)

    val_acc  = accuracy_score(y_v,  y_v_pred)
    test_acc = accuracy_score(y_te, y_te_pred)
    val_f1   = f1_score(y_v,  y_v_pred,  average='macro', zero_division=0)
    test_f1  = f1_score(y_te, y_te_pred, average='macro', zero_division=0)
    logger.info(f"  [Metrics] Val  — Acc: {val_acc:.4f}  Macro-F1: {val_f1:.4f}")
    logger.info(f"  [Metrics] Test — Acc: {test_acc:.4f}  Macro-F1: {test_f1:.4f}")

    # ── Per-class breakdown (test set) ────────────────────────────────────────
    all_labels = sorted(set(y_te) | set(y_te_pred))
    report = classification_report(
        y_te, y_te_pred,
        labels=all_labels,
        zero_division=0,
        output_dict=True,
    )
    width = 72
    sep    = f"+{'-' * (width - 2)}+"
    header = f"|{'  Per-Class Metrics (Test Set) ':─^{width - 2}}|"
    col_hdr = f"|  {'Emotion':<18}  {'F1':>6}  {'Bar':<14}  {'Prec':>6}  {'Recall':>6}  {'n':>5}  |"
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    logger.info(col_hdr)
    logger.info(sep)
    for lbl in EMOTION_LABELS:
        r = report.get(lbl, {})
        support = int(r.get("support", 0))
        f1   = r.get("f1-score",  0.0)
        prec = r.get("precision", 0.0)
        rec  = r.get("recall",    0.0)
        bar  = _bar(f1, width=14)
        tag  = "  " if support > 0 else "⚠ "
        logger.info(
            f"|  {tag}{lbl:<18}  {f1:>6.3f}  {bar:<14}  {prec:>6.3f}  {rec:>6.3f}  {support:>5}  |"
        )
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    macro_p  = report.get("macro avg", {}).get("precision", 0.0)
    macro_r  = report.get("macro avg", {}).get("recall", 0.0)
    logger.info(sep)
    logger.info(
        f"|  {'  macro avg':<20}  {macro_f1:>6.3f}  {'':14}  {macro_p:>6.3f}  {macro_r:>6.3f}  {'':>5}  |"
    )
    logger.info(sep)

    # Log mean gate weights on test set for monitoring (expert collapse detection)
    alpha_te = wrapper.get_gate_weights(X_te_arr)
    alpha_means = alpha_te.mean(axis=0)
    ctx_log = f"  ctx:{alpha_means[3]:.3f}" if len(alpha_means) > 3 else ""
    logger.info(
        f"  [Gate α] Test mean — "
        f"vader:{alpha_means[0]:.3f}  bert:{alpha_means[1]:.3f}  goe:{alpha_means[2]:.3f}{ctx_log}"
    )

    # ── Temperature calibration (Platt scaling) ──────────────────────────────
    calibration_temperature = 1.0
    try:
        from scipy.optimize import minimize_scalar
        y_v_idx = np.array([EMOTION_LABELS.index(y) if y in EMOTION_LABELS else 0 for y in y_v])
        proba_v = wrapper.predict_proba(X_v_arr)
        log_p   = np.log(np.clip(proba_v, 1e-8, 1.0))

        def nll(T):
            scaled = log_p / max(T, 0.1)
            scaled -= scaled.max(axis=1, keepdims=True)
            exp_s  = np.exp(scaled)
            p_norm = exp_s / exp_s.sum(axis=1, keepdims=True)
            return -np.mean(np.log(p_norm[np.arange(len(y_v_idx)), y_v_idx] + 1e-8))

        result = minimize_scalar(nll, bounds=(0.5, 5.0), method='bounded')
        calibration_temperature = float(result.x)
        logger.info(f"  [Calibration] Temperature T={calibration_temperature:.4f}")
    except Exception as e:
        logger.warning(f"  [Calibration] Failed: {e} — using T=1.0")

    phase_times["Evaluation"] = time.time() - t_eval
    logger.info(f"  ⏱ Evaluation complete in {phase_times['Evaluation']:.1f}s")

    # ── Accuracy gate + deploy ─────────────────────────────────────────────────
    t_deploy = time.time()
    deployed = test_acc >= ACCURACY_GATE

    if deployed:
        tmp = MODEL_PATH.with_suffix(".tmp.pkl")
        # Move to CPU before pickling — the pkl must be portable to Docker (Linux/CPU)
        # even when training ran on MPS or CUDA.
        wrapper.model_.cpu()
        wrapper._device = torch.device("cpu")
        with open(tmp, "wb") as f:
            pickle.dump(wrapper, f)
        tmp.rename(MODEL_PATH)

        with open(META_PATH, "w") as f:
            json.dump({
                "trained_at":          datetime.datetime.utcnow().isoformat() + "Z",
                "model_version":       "v2-gating-ensemble",
                "training_mode":       "bootstrap" if is_bootstrap else "continuous",
                "training_samples":    len(X_tr),
                "filtered_samples":    n_filtered,
                "validation_accuracy": round(val_acc,  4),
                "validation_f1_macro": round(val_f1,   4),
                "test_accuracy":       round(test_acc, 4),
                "test_f1_macro":       round(test_f1,  4),
                "previous_accuracy":   round(prev_meta.get("test_accuracy", 0), 4),
                "improvement":         round(test_acc - prev_meta.get("test_accuracy", 0), 4),
                "accuracy_gate":       ACCURACY_GATE,
                "calibration_temperature": round(calibration_temperature, 4),
                "gate_weights_mean":   {
                    "vader":   round(float(alpha_means[0]), 4),
                    "bert":    round(float(alpha_means[1]), 4),
                    "goe":     round(float(alpha_means[2]), 4),
                    **( {"context": round(float(alpha_means[3]), 4)} if len(alpha_means) > 3 else {} ),
                },
            }, f, indent=2)

        if reload_callback is not None:
            reload_callback(wrapper)

        ready_marker = MODEL_PATH.parent / ".ready"
        ready_marker.touch()

        # Publish model stats to Redis for API health endpoint consumption
        try:
            _r = redis_sync.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True,
            )
            _stats = {
                "test_accuracy":         str(round(test_acc, 4)),
                "test_f1_macro":         str(round(test_f1, 4)),
                "val_f1_macro":          str(round(val_f1, 4)),
                "ctx_gate":              str(round(float(alpha_means[3]), 4)) if len(alpha_means) > 3 else "0",
                "goe_gate":              str(round(float(alpha_means[2]), 4)),
                "vader_gate":            str(round(float(alpha_means[0]), 4)),
                "bert_gate":             str(round(float(alpha_means[1]), 4)),
                "calibration_temperature": str(round(calibration_temperature, 4)),
                "feature_dim":           str(FEATURE_DIM),
                "model_version":         "v2-gating-ensemble-bayesian",
                "training_samples":      str(len(X_tr)),
                "last_trained_utc":      datetime.datetime.utcnow().isoformat() + "Z",
                "status":                "ready",
            }
            _r.hset("model:stats", mapping=_stats)
            _r.expire("model:stats", 86400 * 7)
        except Exception as _re:
            logger.debug(f"Could not write model stats to Redis: {_re}")

        try:
            _r.publish(RELOAD_CHANNEL, json.dumps({
                "model_path":    str(MODEL_PATH),
                "test_accuracy": round(test_acc, 4),
                "trained_at":    datetime.datetime.utcnow().isoformat() + "Z",
            }))
        except Exception as _pe:
            logger.warning(f"Could not publish model_reload_signal to Redis: {_pe}")

        if not hasattr(start_trainer_thread, '_initial_trained'):
            logger.info("Training complete. Opening system gates.")
            setattr(start_trainer_thread, '_initial_trained', True)

    phase_times["Deploy / gate"] = time.time() - t_deploy

    # ── Cycle timing summary ──────────────────────────────────────────────────
    total_cycle = time.time() - cycle_start
    if total_cycle >= 60:
        total_cycle_str = f"{int(total_cycle // 60)}m {int(total_cycle % 60)}s"
    else:
        total_cycle_str = f"{total_cycle:.1f}s"
    phase_times["─── Total Cycle ───"] = ""
    phase_times["  Wall-clock time"] = total_cycle_str

    # Format all phase times for the summary table
    phase_summary = {}
    for k, v in phase_times.items():
        if isinstance(v, float):
            if v >= 60:
                phase_summary[k] = f"{int(v // 60)}m {int(v % 60)}s"
            else:
                phase_summary[k] = f"{v:.1f}s"
        else:
            phase_summary[k] = v
    logger.log_stats("⏱ Cycle Timing Breakdown", phase_summary)

    print_report(prev_meta, test_acc, test_f1, len(X_tr), n_filtered, deployed, dataset_composition)


# ── Background worker ───────────────────────────────────────────────────────────

def start_trainer_thread(reload_callback) -> threading.Thread:
    """
    Spawn a daemon thread that runs run_one_cycle() every RETRAIN_INTERVAL seconds.
    reload_callback(wrapper) is called after each successful deploy so main.py
    can hot-swap META_LEARNER without restarting the container.
    """
    def _loop():
        logger.info("Trainer started.")
        logger.log_stats("Trainer Configuration", {
            "RETRAIN_INTERVAL_SECONDS": RETRAIN_INTERVAL,
            "ACCURACY_GATE":            ACCURACY_GATE,
            "MAX_EMPATHETIC_SAMPLES":   MAX_EMPATHETIC_SAMPLES,
            "MIN_DB_SAMPLES":           MIN_DB_SAMPLES,
            "MODEL_PATH":               str(MODEL_PATH),
        })
        logger.info("API access will be enabled on first model completion.")

        r = None
        try:
            r = redis_sync.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True,
            )
        except Exception as e:
            logger.warning(f"Trainer cannot connect to Redis: {e}")

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
