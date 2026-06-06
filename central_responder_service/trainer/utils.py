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
    results = []
    total = len(texts)
    checkpoints = {int(total * p) for p in (0.25, 0.50, 0.75)} if total > 100 else set()
    for i in range(0, total, batch_size):
        chunk = [t[:512] for t in texts[i:i + batch_size]]
        try:
            out = model(chunk)
            for item in out:
                results.append({r['label']: r['score'] for r in item})
        except Exception:
            results.extend([{} for _ in chunk])
        done = len(results)
        if any(done >= cp for cp in list(checkpoints)):
            pct = int(done / total * 100)
            tag = f"[{label}] " if label else ""
            logger.info(f"  {tag}NLP progress: {done}/{total} ({pct}%)")
            checkpoints = {cp for cp in checkpoints if cp > done}
    return results


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
