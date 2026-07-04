import os
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


from shared.constants import EMOTION_LABELS
from shared.utils.logger import get_logger

logger = get_logger("trainer")


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


_TRAINER_PROTO_SENTENCES = [
    "I truly admire and respect this person, it is deeply impressive",
    "This is hilarious and funny, I cannot stop laughing",
    "I am furious and extremely angry, this enrages me",
    "This is so annoying and irritating to deal with",
    "I approve of this decision and think it is the right choice",
    "I care deeply about this and want to help and support",
    "I am confused and do not understand what is happening here",
    "I am curious and want to explore and learn more about this",
    "I want this so badly and desire it deeply",
    "I am deeply disappointed and let down by what happened",
    "I strongly disapprove of this and think it is wrong",
    "This disgusts me and makes me feel revolted and sick",
    "I feel embarrassed and ashamed about what occurred",
    "I am incredibly excited and thrilled, this is amazing",
    "I am terrified and scared, this fills me with fear and dread",
    "I am very grateful and thankful for this kindness",
    "I am overwhelmed with grief and deep sorrow and loss",
    "I feel joyful and happy, this fills me with pure happiness",
    "I feel deep love and affection for this person",
    "I feel nervous and anxious and worried about what might happen",
    "I feel optimistic and hopeful that things will work out well",
    "I feel proud of this great achievement and accomplishment",
    "I just realized something important that I had not noticed before",
    "I feel such relief that this difficult situation resolved well",
    "I feel deep remorse and regret for what I said or did",
    "I feel very sad and melancholy, my heart is heavy with sorrow",
    "I am completely surprised and shocked by this unexpected event",
    "This is a normal everyday situation with no particular emotional response",
]


def _get_analyzers(device):
    logger.info("Loading analyzers transiently into RAM...")

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import pipeline as hf_pipeline

    vader = SentimentIntensityAnalyzer()
    bert  = hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=device,
    )
    _finetuned = str(Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")).parent / "samlowe_finetuned")
    _goe_primary_model = _finetuned if os.path.isdir(_finetuned) else "SamLowe/roberta-base-go_emotions"
    logger.info(f"Loading GoEmotions primary model ({_goe_primary_model})...")
    goe_primary = hf_pipeline(
        "text-classification",
        model=_goe_primary_model,
        top_k=None,
        device=device,
    )
    goe_secondary = hf_pipeline(
        "text-classification",
        model="bhadresh-savani/bert-base-go-emotion",
        top_k=None,
        device=device,
    )

    logger.info("Loading sentence transformer for semantic NLI (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    sent_model = SentenceTransformer("all-MiniLM-L6-v2")
    proto_embs = sent_model.encode(_TRAINER_PROTO_SENTENCES, convert_to_numpy=True)
    proto_norms = np.linalg.norm(proto_embs, axis=1, keepdims=True) + 1e-8
    proto_norm  = proto_embs / proto_norms

    # Wrap into an ensemble callable matching the pipeline interface
    class _GoEEnsemble:
        def __init__(self):
            self.model = goe_primary.model

        def __call__(self, texts, batch_size=32):
            if isinstance(texts, str):
                texts = [texts]
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_primary   = pool.submit(goe_primary,   texts, batch_size=batch_size)
                f_secondary = pool.submit(goe_secondary, texts, batch_size=batch_size)
                p_out = f_primary.result()
                s_out = f_secondary.result()

            sent_embs = sent_model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
            e_norms   = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-8
            sent_norm = sent_embs / e_norms
            sims      = sent_norm @ proto_norm.T
            exp_s     = np.exp((sims - sims.max(axis=1, keepdims=True)) * 5.0)
            sims_prob = exp_s / exp_s.sum(axis=1, keepdims=True)

            results = []
            for idx, (p_item, s_item) in enumerate(zip(p_out, s_out)):
                p_map   = {r["label"]: r["score"] for r in p_item}
                s_map   = {r["label"]: r["score"] for r in s_item}
                sem_map = {e: float(sims_prob[idx, i]) for i, e in enumerate(EMOTION_LABELS)}
                merged  = {
                    lbl: 0.55 * p_map.get(lbl, 0.0) + 0.25 * s_map.get(lbl, 0.0) + 0.20 * sem_map.get(lbl, 0.0)
                    for lbl in set(p_map) | set(s_map) | set(sem_map)
                }
                results.append([{"label": k, "score": v} for k, v in merged.items()])
            return results

        # expose device for GPU-detection logic in _run_parallel_batches
        @property
        def device(self):
            return goe_primary.device

    goe = _GoEEnsemble()
    logger.info("Analyzers fully loaded (GoEmotions: 55% SamLowe + 25% bhadresh + 20% semantic NLI).")
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
            out = model(chunk, batch_size=batch_size)
            for item in out:
                results.append({r['label']: r['score'] for r in item})
        except Exception as e:
            logger.warning(f"  {tag}Batch failed: {e}")
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
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(texts)

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
                out = model(chunk, batch_size=batch_size)
                for item in out:
                    results.append({r['label']: r['score'] for r in item})
            except Exception as e:
                logger.warning(f"  [{label}] Batch failed: {e}")
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
                
                if getattr(model.model, "device", None) is not None and model.model.device.type == "mps":
                    import torch
                    torch.mps.empty_cache()
                    
        return label, results

    bert_label = f"{label_prefix}/BERT" if label_prefix else "BERT"
    goe_label  = f"{label_prefix}/GoE"  if label_prefix else "GoE"

    bert_results = None
    goe_results  = None

    is_gpu = bert_model.device.type != "cpu"

    if is_gpu:
        logger.info(f"  [{label_prefix}] GPU detected — Running sequentially to prevent lock contention.")
        _, bert_results = _run_hf(bert_model, bert_label)
        _, goe_results  = _run_hf(goe_model, goe_label)
    else:
        logger.info(f"  [{label_prefix}] CPU detected — Running concurrently via threads.")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(_run_hf, bert_model, bert_label): bert_label,
                pool.submit(_run_hf, goe_model, goe_label): goe_label,
            }
            for future in as_completed(futures):
                label, results = future.result()
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


def filter_outliers(X: list, y: list, goe_list: list, has_cdm: list = None):
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
