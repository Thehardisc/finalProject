"""
trainer.py — Periodic retraining background thread for central_responder_service.

Runs in a daemon thread alongside the main Redis consumer loop.
After each successful cycle, hot-reloads the model into the global META_LEARNER
variable so predictions immediately use the new weights — no restart needed.

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

# ── Config ────────────────────────────────────────────────────────────────────
RETRAIN_INTERVAL = int(os.environ.get("RETRAIN_INTERVAL_SECONDS", 600))
ACCURACY_GATE    = float(os.environ.get("ACCURACY_GATE", 0.40))
MAX_SAMPLES      = int(os.environ.get("MAX_SAMPLES", 10000))
MODEL_PATH       = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
META_PATH        = MODEL_PATH.with_name("meta_weights_meta.json")

# Label space — mirrors meta_learner.py
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
VADER_KEYS  = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
BERT_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CLI Reporter                                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _bar(value: float, width: int = 25) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)

def print_report(prev_meta: dict, new_acc: float, n_train: int,
                 n_filtered: int, deployed: bool):
    prev_acc = prev_meta.get("test_accuracy")
    delta    = (new_acc - prev_acc) if prev_acc is not None else None
    ts       = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    W = 58

    print("\n" + "╔" + "═"*(W-2) + "╗")
    print(f"║{'  🔄  Retraining Report':^{W-2}}║")
    print("╠" + "═"*(W-2) + "╣")
    print(f"║  {ts:<{W-4}}║")
    print(f"║  Samples trained : {n_train}  │  Filtered (bad): {n_filtered:<{W-40}}║")
    print("╠" + "═"*(W-2) + "╣")

    if prev_acc is not None:
        print(f"║  Previous : {prev_acc:.4f}  {_bar(prev_acc):<25} ║")
    else:
        print(f"║  Previous : — (no prior model){'':<{W-32}}║")

    print(f"║  New      : {new_acc:.4f}  {_bar(new_acc):<25} ║")

    if delta is not None:
        sign, emoji = ("+", "📈") if delta >= 0 else ("", "📉")
        print(f"║  Delta    : {emoji} {sign}{delta*100:.2f}%{'':<{W-26}}║")

    print("╠" + "═"*(W-2) + "╣")
    if deployed:
        print(f"║  {'✅  DEPLOYED  — model hot-reloaded in memory':<{W-4}}║")
    else:
        reason = (f"accuracy {new_acc:.4f} < gate {ACCURACY_GATE:.2f}"
                  if new_acc < ACCURACY_GATE else "accuracy regressed")
        print(f"║  {'❌  REJECTED  — ' + reason:<{W-4}}║")
    print("╚" + "═"*(W-2) + "╝\n")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Feature Builder                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def build_fv(vader: dict, bert: dict, goe: dict, emoji: dict) -> np.ndarray:
    vec = []
    for k in VADER_KEYS:  vec.append(float(vader.get(k, 0.0)))
    for k in BERT_LABELS: vec.append(float(bert.get(k,  0.0)))
    for k in EMOTION_LABELS: vec.append(float(goe.get(k,   0.0)))
    for k in EMOTION_LABELS: vec.append(float(emoji.get(k,  0.0)))
    return np.array(vec, dtype=np.float32)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Bad-Data Filters                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

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
        print(f"  [Filter] Balance cap: removed {removed} samples (cap={cap}/class).")
    return cX, cy


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Analyzer Helpers                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _get_analyzers(device):
    print("[Trainer] Loading analyzers transiently into RAM...")
    import torch
    
    # Constrain threads to prevent PyTorch from inflating RAM overhead during inference
    torch.set_num_threads(1)
    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import pipeline as hf_pipeline
    vader = SentimentIntensityAnalyzer()
    bert  = hf_pipeline("text-classification",
                          model="j-hartmann/emotion-english-distilroberta-base",
                          return_all_scores=True,
                          device=device)
    goe   = hf_pipeline("text-classification",
                          model="SamLowe/roberta-base-go_emotions",
                          return_all_scores=True,
                          device=device)
                          
    print("[Trainer] Analyzers fully loaded.")
    return vader, bert, goe

def _vader(v, text):
    s = v.polarity_scores(text)
    return {k: s[k] for k in ['neg','neu','pos','compound']}

def _run(model, text):
    try:    return {r['label']: r['score'] for r in model(text[:512])[0]}
    except: return {}

def _emojinet(text):
    try:
        import emoji as emoji_lib
        DB = {
            "😂":{"joy":0.9,"amusement":0.95}, "😭":{"sadness":0.8,"grief":0.6,"joy":0.2},
            "😍":{"love":0.95,"admiration":0.9,"joy":0.8}, "🔥":{"excitement":0.9,"admiration":0.8,"joy":0.7},
            "💀":{"amusement":0.9,"joy":0.7,"fear":0.1}, "🙃":{"amusement":0.4,"annoyance":0.6,"confusion":0.3},
            "🤔":{"curiosity":0.8,"confusion":0.4,"disapproval":0.2}, "🙄":{"annoyance":0.9,"disapproval":0.8,"disgust":0.5},
            "💩":{"disgust":0.8,"amusement":0.5,"annoyance":0.4}, "❤️":{"love":1.0,"caring":0.9,"joy":0.8},
            "✨":{"excitement":0.7,"admiration":0.6,"joy":0.5},
        }
        found = emoji_lib.distinct_emoji_list(text)
        scores, count = {}, 0
        for ch in found:
            entry = DB.get(ch) or DB.get(ch.replace('\ufe0f',''))
            if entry:
                for e, sc in entry.items(): scores[e] = scores.get(e, 0.0) + sc
                count += 1
        return {k: v/count for k,v in scores.items()} if count else {}
    except: return {}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Training Cycle                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def run_one_cycle(reload_callback):
    """
    Full training + filter + gate + deploy cycle.
    Calls reload_callback(new_model) on success so main.py can hot-swap
    the global META_LEARNER without restarting the container.
    """
    print(f"\n[Trainer] ═══ Starting cycle at {datetime.datetime.utcnow().strftime('%H:%M:%S UTC')} ═══")

    prev_meta = {}
    if META_PATH.exists():
        try:
            with open(META_PATH) as f: prev_meta = json.load(f)
        except Exception: pass

    # Load dataset
    print("[Trainer] Loading GoEmotions dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    except Exception as e:
        print(f"[Trainer] ❌ Dataset load failed: {e}")
        return

    train_raw = list(ds["train"])[:MAX_SAMPLES]
    val_raw   = list(ds["validation"])[:MAX_SAMPLES // 5]
    test_raw  = list(ds["test"])[:MAX_SAMPLES // 5]

    # Setup device mapping and load analyzers
    import torch
    device = 0 if torch.cuda.is_available() else -1
    vader, bert, goe = _get_analyzers(device)

    def process(split, name):
        X, y, gs_list = [], [], []
        for i, s in enumerate(split):
            if i % 500 == 0: print(f"  [Trainer] {name} {i}/{len(split)}", end="\r")
            lids = s.get("labels", [])
            if not lids or lids[0] >= len(EMOTION_LABELS): continue
            text = s["text"]
            vs = {f"vader_{k}": v for k, v in _vader(vader, text).items()}
            bs = _run(bert, text)
            gs = _run(goe,  text)
            es = _emojinet(text)
            gs_list.append(gs)
            X.append(build_fv(vs, bs, gs, es))
            y.append(EMOTION_LABELS[lids[0]])
        print(f"  [Trainer] {name}: {len(X)} samples.          ")
        return X, y, gs_list

    print("[Trainer] Building feature vectors...")
    X_tr, y_tr, gs_tr = process(train_raw, "train")
    X_v,  y_v,  _     = process(val_raw,   "val  ")
    X_te, y_te, _     = process(test_raw,  "test ")

    # Aggressive memory cleanup immediately after features are computed limits RAM peak
    del vader, bert, goe
    import gc
    gc.collect()
    print("[Trainer] Aggressively purged transient analyzers from RAM.")

    n_before = len(X_tr)
    print("[Trainer] Applying bad-data filters...")

    # Layer 1 already enforced (only valid EMOTION_LABELS pass process())
    X_tr, y_tr, n_out = filter_outliers(X_tr, y_tr, gs_tr)
    print(f"  Layer 2 (outlier filter)  : removed {n_out}")
    X_tr, y_tr        = filter_balance(X_tr, y_tr)
    n_filtered = n_before - len(X_tr)

    if len(X_tr) < 100:
        print("[Trainer] ❌ Too few samples after filtering. Skipping.")
        return

    # Train
    print("[Trainer] Training Logistic Regression...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                    multi_class='multinomial', class_weight='balanced',
                                    random_state=42))
    ])
    pipeline.fit(np.array(X_tr), y_tr)
    val_acc  = accuracy_score(y_v,  pipeline.predict(np.array(X_v)))
    test_acc = accuracy_score(y_te, pipeline.predict(np.array(X_te)))

    # Layer 4: accuracy gate
    deployed = test_acc >= ACCURACY_GATE

    if deployed:
        # Atomic write: tmp → rename so central_responder never reads partial file
        tmp = MODEL_PATH.with_suffix(".tmp.pkl")
        with open(tmp, 'wb') as f: pickle.dump(pipeline, f)
        tmp.rename(MODEL_PATH)

        with open(META_PATH, 'w') as f:
            json.dump({
                "trained_at":         datetime.datetime.utcnow().isoformat() + "Z",
                "training_samples":   len(X_tr),
                "filtered_samples":   n_filtered,
                "validation_accuracy": round(val_acc,  4),
                "test_accuracy":       round(test_acc, 4),
                "previous_accuracy":   round(prev_meta.get("test_accuracy", 0), 4),
                "improvement":         round(test_acc - prev_meta.get("test_accuracy", 0), 4),
                "accuracy_gate":       ACCURACY_GATE,
            }, f, indent=2)

        # Hot-reload — no container restart needed
        reload_callback(pipeline)

    print_report(prev_meta, test_acc, len(X_tr), n_filtered, deployed)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Daemon Thread Entry Point                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def start_trainer_thread(reload_callback):
    """
    Spawn a daemon thread that runs run_one_cycle() every RETRAIN_INTERVAL seconds.
    The reload_callback(new_model) is called after each successful deploy so
    main.py can swap the global META_LEARNER without restarting.
    """
    def _loop():
        print(f"[Trainer] 🚀 Started. Interval={RETRAIN_INTERVAL}s, Gate={ACCURACY_GATE}, Samples={MAX_SAMPLES}")
        while True:
            try:
                run_one_cycle(reload_callback)
            except Exception as e:
                print(f"[Trainer] ❌ Unhandled error: {e}")
                traceback.print_exc()
            print(f"[Trainer] Sleeping {RETRAIN_INTERVAL}s...")
            time.sleep(RETRAIN_INTERVAL)

    t = threading.Thread(target=_loop, name="trainer", daemon=True)
    t.start()
    return t
