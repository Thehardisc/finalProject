"""
boost_train.py — Fast standalone retraining on GoEmotions balanced cache.

Skips all NLP inference. Loads the pre-built 107-dim feature vectors from
.cache/goemotions_direct_cache.pkl (12K samples, 500/class × 24 classes),
splits 70/15/15, trains GatingEnsembleNet with MPS acceleration, saves to
.cache/meta_weights.pkl.

Usage:
    cd <project_root>
    source .venv/bin/activate
    python3 boost_train.py
"""

import datetime
import json
import os
import pickle
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "central_responder_service"))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MODEL_PATH", str(ROOT / ".cache" / "meta_weights.pkl"))

from shared.constants import EMOTION_LABELS, FEATURE_DIM
from trainer.models import train_gating_network


def main():
    CACHE_PATH = ROOT / ".cache" / "goemotions_direct_cache.pkl"
    MODEL_PATH = Path(os.environ["MODEL_PATH"])
    META_PATH  = MODEL_PATH.with_name("meta_weights_meta.json")
    CACHE_KEY  = f"goemotions_direct_v3_500_{FEATURE_DIM}"

    print(f"[boost_train] Loading GoEmotions cache: {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)

    assert cached.get("cache_key") == CACHE_KEY, (
        f"Cache key mismatch: {cached.get('cache_key')} != {CACHE_KEY}. "
        f"Delete .cache/goemotions_direct_cache.pkl and re-run to rebuild."
    )

    X_raw, y_raw, _ = cached["data"]
    X_arr = np.array(X_raw, dtype=np.float32)
    y_arr = list(y_raw)
    print(f"[boost_train] Loaded {len(y_arr)} samples, feature_dim={X_arr.shape[1]}")

    counts = Counter(y_arr)
    print(f"[boost_train] {len(counts)} distinct classes, min={min(counts.values())}, max={max(counts.values())}")

    # 70/15/15 stratified split
    X_tr, X_hold, y_tr, y_hold = train_test_split(
        X_arr, y_arr, test_size=0.30, random_state=42, stratify=y_arr
    )
    X_v, X_te, y_v, y_te = train_test_split(
        X_hold, y_hold, test_size=0.50, random_state=42, stratify=y_hold
    )
    print(f"[boost_train] Split: {len(y_tr)} train / {len(y_v)} val / {len(y_te)} test")

    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[boost_train] Training on device={device}")

    has_cdm = np.zeros(len(y_tr), dtype=bool)

    wrapper = train_gating_network(
        X_tr=X_tr,
        y_tr=y_tr,
        X_v=X_v,
        y_v=y_v,
        classes=EMOTION_LABELS,
        has_cdm=has_cdm,
        n_epochs=120,
        batch_size=512,
        lr=5e-4,
        weight_decay=1e-4,
        load_balance_coeff=1e-3,
        patience=20,
        device=device,
    )

    # Evaluate
    y_te_pred = wrapper.predict(X_te)
    y_v_pred  = wrapper.predict(X_v)

    val_acc  = accuracy_score(y_v,  y_v_pred)
    test_acc = accuracy_score(y_te, y_te_pred)
    val_f1   = f1_score(y_v,  y_v_pred,  average='macro', zero_division=0)
    test_f1  = f1_score(y_te, y_te_pred, average='macro', zero_division=0)

    print(f"\n[boost_train] ── Results ──")
    print(f"  Val  Acc={val_acc:.4f}  Macro-F1={val_f1:.4f}")
    print(f"  Test Acc={test_acc:.4f}  Macro-F1={test_f1:.4f}")
    print("\n" + classification_report(y_te, y_te_pred, zero_division=0))

    import torch as _t
    X_te_s = wrapper.scaler_.transform(X_te)
    X_te_t = _t.tensor(X_te_s, dtype=_t.float32, device="cpu")
    wrapper.model_.cpu()
    wrapper.model_.eval()
    with _t.no_grad():
        _, alpha = wrapper.model_(X_te_t)
    alpha_means = alpha.mean(dim=0).numpy()
    print(
        f"[boost_train] Gate α: vader={alpha_means[0]:.3f}  bert={alpha_means[1]:.3f}  "
        f"goe={alpha_means[2]:.3f}  ctx={alpha_means[3]:.3f}"
    )

    ACCURACY_GATE = float(os.environ.get("ACCURACY_GATE", "0.40"))
    if test_acc < ACCURACY_GATE:
        print(f"[boost_train] test_acc={test_acc:.4f} below gate={ACCURACY_GATE}. NOT saving.")
        return

    prev_acc = 0.0
    if META_PATH.exists():
        try:
            prev_acc = json.loads(META_PATH.read_text()).get("test_accuracy", 0.0)
        except Exception:
            pass

    wrapper.model_.cpu()
    wrapper._device = _t.device("cpu")

    tmp = MODEL_PATH.with_suffix(".tmp.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(wrapper, f)
    tmp.rename(MODEL_PATH)
    print(f"[boost_train] Model saved → {MODEL_PATH}")

    meta = {
        "trained_at":          datetime.datetime.utcnow().isoformat() + "Z",
        "model_version":       "v2-gating-ensemble",
        "training_mode":       "boost-goemotions",
        "training_samples":    len(y_tr),
        "filtered_samples":    0,
        "validation_accuracy": round(val_acc,  4),
        "validation_f1_macro": round(val_f1,   4),
        "test_accuracy":       round(test_acc, 4),
        "test_f1_macro":       round(test_f1,  4),
        "previous_accuracy":   round(prev_acc, 4),
        "improvement":         round(test_acc - prev_acc, 4),
        "accuracy_gate":       ACCURACY_GATE,
        "calibration_temperature": 1.0,
        "gate_weights_mean": {
            "vader":   round(float(alpha_means[0]), 4),
            "bert":    round(float(alpha_means[1]), 4),
            "goe":     round(float(alpha_means[2]), 4),
            "context": round(float(alpha_means[3]), 4),
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[boost_train] Meta saved → {META_PATH}")
    (MODEL_PATH.parent / ".ready").touch()

    print(f"\n[boost_train] Done. Acc: {prev_acc:.4f} → {test_acc:.4f} (+{test_acc-prev_acc:+.4f})")
    print(
        f"[boost_train] To reload in Docker:\n"
        f"  docker exec finalproject-main-central_responder_service-1 "
        f"redis-cli PUBLISH model_reload_signal "
        f"'{{\"test_accuracy\": {test_acc:.4f}, \"trained_at\": \"now\"}}'"
    )


if __name__ == "__main__":
    main()
