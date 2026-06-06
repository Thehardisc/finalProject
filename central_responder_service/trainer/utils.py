"""
trainer/utils.py — Reporting, NLP analyser helpers, and data-filtering utilities.
"""

import statistics
from collections import Counter

import os

import numpy as np
import torch

from shared.constants import EMOTION_LABELS
from shared.utils.logger import get_logger

logger = get_logger("trainer")


# ── Reporting ───────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 25) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def print_report(
    prev_meta: dict,
    new_acc: float,
    new_f1: float,
    n_train: int,
    n_filtered: int,
    deployed: bool,
    dataset_composition=None,
) -> None:
    prev_acc = prev_meta.get("test_accuracy")
    delta    = (new_acc - prev_acc) if prev_acc is not None else None

    stats = {
        "Previous Accuracy":   f"{prev_acc:.4f}" if prev_acc is not None else "N/A",
        "New Accuracy (test)": f"{new_acc:.4f}  {_bar(new_acc)}",
        "New F1 (macro)":      f"{new_f1:.4f}  {_bar(new_f1)}",
        "Samples Trained":     n_train,
        "Samples Filtered":    n_filtered,
        "Deployment":          "✅ DEPLOYED" if deployed else "❌ REJECTED (accuracy/regression)",
    }
    if delta is not None:
        direction    = "↑" if delta >= 0 else "↓"
        stats["Delta"] = f"{direction} {delta*100:+.2f}%"

    if dataset_composition:
        stats["─── Dataset Sources ───"] = ""
        for src, count in dataset_composition.items():
            stats[f"  {src}"] = count

    logger.log_stats("Retraining Report", stats)


# ── NLP analyser utilities ──────────────────────────────────────────────────────

def _get_analyzers(device):
    logger.info("Loading analyzers transiently into RAM...")

    cpu_count = os.cpu_count() or 4
    on_gpu = device != -1  # CUDA (int >= 0) or MPS (str)
    if on_gpu:
        n_threads = max(2, cpu_count // 4)
        logger.info(f"GPU device — setting {n_threads} PyTorch CPU threads ({cpu_count} cores available)")
    else:
        n_threads = max(1, cpu_count - 1)
        logger.info(f"CPU-only — setting {n_threads} PyTorch threads ({cpu_count} cores available)")
    torch.set_num_threads(n_threads)

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import pipeline as hf_pipeline

    vader = SentimentIntensityAnalyzer()
    bert  = hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=device,
    )
    goe   = hf_pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        device=device,
    )
    logger.info("Analyzers fully loaded.")
    return vader, bert, goe


def _vader(v, text: str) -> dict:
    s = v.polarity_scores(text)
    return {k: s[k] for k in ['neg', 'neu', 'pos', 'compound']}


def _run(model, text: str) -> dict:
    try:
        return {r['label']: r['score'] for r in model(text[:512])[0]}
    except Exception:
        return {}


def _run_batch(model, texts: list, batch_size: int = 32, label: str = "") -> list:
    """Run a HuggingFace pipeline on a list of texts in batches. Returns one dict per text."""
    import time as _time
    results = []
    total = len(texts)
    t0 = _time.time()
    last_logged = 0
    tag = f"[{label}] " if label else ""
    logger.info(f"  {tag}Starting NLP batch — {total} texts, batch_size={batch_size}")
    for i in range(0, total, batch_size):
        chunk = [t[:512] for t in texts[i:i + batch_size]]
        try:
            out = model(chunk)
            for item in out:
                results.append({r['label']: r['score'] for r in item})
        except Exception:
            results.extend([{} for _ in chunk])
        done = len(results)
        if done - last_logged >= 100 or done == total:
            elapsed = _time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            remaining = total - done
            eta_sec = remaining / speed if speed > 0 else 0
            if eta_sec >= 60:
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
            else:
                eta_str = f"{eta_sec:.1f}s"
            pct = int(done / total * 100)
            logger.info(
                f"  {tag}NLP progress: {done}/{total} ({pct}%)  "
                f"{speed:.1f} samples/s  ETA={eta_str}"
            )
            last_logged = done
    elapsed_total = _time.time() - t0
    speed_total = total / elapsed_total if elapsed_total > 0 else 0
    if elapsed_total >= 60:
        elapsed_str = f"{int(elapsed_total // 60)}m {int(elapsed_total % 60)}s"
    else:
        elapsed_str = f"{elapsed_total:.1f}s"
    logger.info(f"  {tag}NLP complete — {total} texts in {elapsed_str} ({speed_total:.1f} samples/s)")
    return results


def _run_parallel_batches(
    bert_model, goe_model, texts: list,
    vader_analyzer=None,
    batch_size: int = 32, label_prefix: str = "",
) -> tuple:
    """
    Run BERT and GoEmotions concurrently using threads, with per-100-sample
    progress logging from each model. VADER runs inline first (it's rule-based
    and finishes in <1s).

    HuggingFace pipelines release the GIL during the C++/CUDA forward pass,
    so threading gives near-perfect overlap on CPU and partial overlap on GPU.

    Returns:
        If vader_analyzer is provided: (vader_results, bert_results, goe_results)
        Otherwise: (bert_results, goe_results)
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(texts)

    # ── VADER: run inline (rule-based, <1s for 20k texts) ─────────────────
    vader_results = None
    if vader_analyzer:
        t_v = _time.time()
        vader_results = []
        for t in texts:
            try:
                vader_results.append({f"vader_{k}": v for k, v in _vader(vader_analyzer, t).items()})
            except Exception:
                vader_results.append({})
        v_elapsed = _time.time() - t_v
        v_speed = total / v_elapsed if v_elapsed > 0 else 0
        logger.info(
            f"  [{label_prefix}/VADER] Done — {total} texts in {v_elapsed:.1f}s "
            f"({v_speed:.0f} samples/s)"
        )

    # ── BERT + GoEmotions: run in parallel with progress logging ──────────
    logger.info(
        f"  [{label_prefix}] Running BERT + GoEmotions in parallel — "
        f"{total} texts, batch_size={batch_size}"
    )
    t0 = _time.time()

    def _run_hf(model, label):
        results = []
        last_logged = 0
        t_start = _time.time()
        for i in range(0, total, batch_size):
            chunk = [t[:512] for t in texts[i:i + batch_size]]
            try:
                out = model(chunk)
                for item in out:
                    results.append({r['label']: r['score'] for r in item})
            except Exception:
                results.extend([{} for _ in chunk])
            done = len(results)
            if done - last_logged >= 100:
                elapsed = _time.time() - t_start
                speed = done / elapsed if elapsed > 0 else 0
                remaining = total - done
                eta_sec = remaining / speed if speed > 0 else 0
                if eta_sec >= 60:
                    eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
                else:
                    eta_str = f"{eta_sec:.1f}s"
                pct = int(done / total * 100)
                logger.info(
                    f"  [{label}] {done}/{total} ({pct}%)  "
                    f"{speed:.1f} samples/s  ETA={eta_str}"
                )
                last_logged = done
        return label, results

    bert_label = f"{label_prefix}/BERT" if label_prefix else "BERT"
    goe_label  = f"{label_prefix}/GoE"  if label_prefix else "GoE"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_run_hf, bert_model, bert_label): bert_label,
            pool.submit(_run_hf, goe_model, goe_label): goe_label,
        }

        bert_results = None
        goe_results  = None
        for future in as_completed(futures):
            label, results = future.result()
            elapsed = _time.time() - t0
            speed = total / elapsed if elapsed > 0 else 0
            if elapsed >= 60:
                elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            else:
                elapsed_str = f"{elapsed:.1f}s"
            logger.info(
                f"  [{label}] ✓ Finished — {total} texts in {elapsed_str} "
                f"({speed:.1f} samples/s)"
            )
            if label == bert_label:
                bert_results = results
            else:
                goe_results = results

    total_elapsed = _time.time() - t0
    if total_elapsed >= 60:
        total_str = f"{int(total_elapsed // 60)}m {int(total_elapsed % 60)}s"
    else:
        total_str = f"{total_elapsed:.1f}s"
    logger.info(
        f"  [{label_prefix}] Parallel NLP complete — {total} texts × 2 models "
        f"in {total_str} (wall-clock)"
    )

    if vader_analyzer:
        return vader_results, bert_results, goe_results
    return bert_results, goe_results


# ── Data filtering ──────────────────────────────────────────────────────────────

def filter_outliers(X: list, y: list, goe_list: list, has_cdm: list = None):
    """Drop samples where GoEmotions gives < 5% confidence to the gold label."""
    cX, cy, cdm, removed = [], [], [], 0
    for i, (fv, label, goe) in enumerate(zip(X, y, goe_list)):
        if label not in EMOTION_LABELS or goe.get(label, 0.0) < 0.05:
            removed += 1
        else:
            cX.append(fv)
            cy.append(label)
            if has_cdm is not None:
                cdm.append(has_cdm[i])
    return (cX, cy, removed, cdm) if has_cdm is not None else (cX, cy, removed)


def filter_balance(X: list, y: list, has_cdm: list = None):
    """Cap any class at 3× the median class count."""
    if not y:
        return (X, y, has_cdm) if has_cdm is not None else (X, y)
    counts = Counter(y)
    cap    = max(50, int(statistics.median(counts.values()) * 3))
    seen   = Counter()
    cX, cy, cdm = [], [], []
    for i, (fv, label) in enumerate(zip(X, y)):
        if seen[label] < cap:
            cX.append(fv)
            cy.append(label)
            if has_cdm is not None:
                cdm.append(has_cdm[i])
            seen[label] += 1
    removed = len(X) - len(cX)
    if removed:
        logger.info(f"  [Filter] Balance cap: removed {removed} samples (cap={cap}/class).")
    return (cX, cy, cdm) if has_cdm is not None else (cX, cy)
