"""Generic loader for local labeled CSVs (columns: text,goemotions_label).

One code path serves every local CSV dataset; adding a new one is a CSV_SETS
entry, not a new module. Current sets: the gold CDM dump (csv_local) plus the
Claude-generated registers written by register_gen.py (hyperbole, banter,
synthetic situational sentences).
"""
import csv
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

from shared.constants import (
    EMOTION_LABELS, FEATURE_DIM, CDM_CTX_DIM, PRIOR_DIM, CONTEXT_DIM,
    N_CDM_STATES, SARCASM_DIM, DYNAMICS_DIM, APPRAISAL_DIM,
)
from shared.utils.logger import get_logger
from meta_learner import build_feature_vector
from trainer.utils import _run_parallel_batches

logger = get_logger("trainer")

_MODEL_DIR = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")).parent
_DATA_DIR  = Path(os.environ.get("TRAINING_DATA_DIR", "/app/training_data"))

# version → cache-key prefix (bump to force recompute); max_rows 0 = unlimited;
# label_caps = per-label row cap; generated = produced by register_gen.py;
# ctx "synthetic" = label-conditioned fake CDM context (else zero context);
# gs "goe" = gold-standard dist is the real GoE output (else one-hot of the label)
CSV_SETS: dict = {
    "csv_local": {
        "tag": "CSVLocal", "version": "csv_local_v2",
        "path": _DATA_DIR / "goemotions_cdm_10k.csv",
        "max_rows": 0, "label_caps": {"neutral": 500}, "generated": False,
        "ctx": None, "gs": "onehot",
    },
    "hyperbole": {
        "tag": "Hyperbole", "version": "hyperbole_v1",
        "path": _MODEL_DIR / "hyperbole_samples.csv",
        "max_rows": int(os.environ.get("HYPERBOLE_MAX_SAMPLES", "500")),
        "label_caps": {}, "generated": True,
        "ctx": None, "gs": "onehot",
    },
    "banter": {
        "tag": "Banter", "version": "banter_v1",
        "path": _MODEL_DIR / "banter_samples.csv",
        "max_rows": int(os.environ.get("BANTER_MAX_SAMPLES", "600")),
        "label_caps": {}, "generated": True,
        "ctx": None, "gs": "onehot",
    },
    "synthetic": {
        "tag": "Synthetic", "version": "synthetic_v1",
        "path": _MODEL_DIR / "synthetic_samples.csv",
        "max_rows": 0, "label_caps": {}, "generated": True,
        "ctx": "synthetic", "gs": "goe",
    },
}


# ── Synthetic-context synthesis ──────────────────────────────────────────────
# Label-conditioned fake CDM context for samples with no real conversation
# history (the "synthetic" set's ctx policy, and db.py's relabeled data).

_LABEL_TO_INTENT: dict = {
    'admiration':     (2,  14,  1),
    'amusement':      (4,  10,  0),
    'approval':       (14,  2, 11),
    'caring':         (12,  1,  9),
    'curiosity':      (10,  3,  0),
    'desire':         (1,  10,  0),
    'excitement':     (2,  14,  4),
    'gratitude':      (14,  1,  2),
    'joy':            (1,   2, 14),
    'love':           (1,  12,  9),
    'optimism':       (14,  1,  2),
    'pride':          (2,  11, 14),
    'relief':         (9,  14,  0),
    'realization':    (0,  10, 11),
    'anger':          (6,   5,  7),
    'annoyance':      (5,  13,  6),
    'disapproval':    (5,  11,  6),
    'disgust':        (6,   5,  0),
    'disappointment': (8,  13,  0),
    'embarrassment':  (8,   0, 13),
    'fear':           (3,   8,  0),
    'grief':          (8,  12,  0),
    'nervousness':    (3,   8, 13),
    'remorse':        (9,   8, 12),
    'sadness':        (8,  12,  0),
    'confusion':      (11,  3,  0),
    'neutral':        (0,  10, 11),
    'surprise':       (4,  10,  0),
}

_STRONG_LABELS = frozenset({
    'anger', 'joy', 'love', 'grief', 'admiration', 'fear',
    'disgust', 'gratitude', 'pride', 'remorse', 'sadness', 'excitement',
})

_LABEL_BASE_VALENCE: dict = {
    'admiration':    0.62,  'amusement':      0.68,  'approval':    0.52,
    'caring':        0.58,  'curiosity':      0.18,  'desire':      0.42,
    'excitement':    0.78,  'gratitude':      0.72,  'joy':         0.82,
    'love':          0.82,  'optimism':       0.62,  'pride':       0.68,
    'realization':   0.12,  'relief':         0.52,  'surprise':    0.12,
    'anger':        -0.72,  'annoyance':     -0.48,  'disapproval': -0.52,
    'disgust':      -0.68,  'disappointment':-0.58,  'embarrassment':-0.42,
    'fear':         -0.62,  'grief':         -0.82,  'nervousness': -0.42,
    'remorse':      -0.58,  'sadness':       -0.68,
    'confusion':    -0.08,  'neutral':        0.00,
}


def build_synthetic_context_vector(
    label: str = None,
    mode: str = "train",
) -> list:
    if mode == "cold":
        return [0.0] * CONTEXT_DIM

    rng = np.random.RandomState()

    if label and label in _LABEL_TO_INTENT:
        primary, secondary, tertiary = _LABEL_TO_INTENT[label]
        if mode == "train" and rng.random() < 0.20:
            cdm_state = int(rng.randint(0, N_CDM_STATES))
        else:
            p = rng.random()
            cdm_state = primary if p < 0.60 else (secondary if p < 0.85 else tertiary)
    else:
        dirichlet_alpha = [3.0] + [1.0] * (N_CDM_STATES - 1)
        cdm_state = int(rng.choice(N_CDM_STATES, p=rng.dirichlet(dirichlet_alpha)))
    cdm_one_hot     = [0.0] * N_CDM_STATES
    cdm_one_hot[cdm_state] = 1.0

    residency  = float(rng.beta(2.0, 5.0))
    transition = [float(rng.randint(0, N_CDM_STATES) / float(N_CDM_STATES)) for _ in range(3)]
    abruptness = float(rng.beta(1.0, 3.0))

    coherence      = float(rng.beta(3.0, 2.0))
    entropy        = float(rng.beta(2.0, 3.0))
    spk_divergence = float(rng.beta(1.0, 4.0))
    acceleration   = float(np.clip(rng.normal(0.0, 0.12), -1.0, 1.0))
    resonance      = float(rng.beta(2.0, 2.0))
    volatility     = float(rng.beta(1.5, 3.0))
    msg_length   = float(rng.uniform(10, 450))
    latency_norm = float(rng.uniform(50, 3000))

    noise_sigma = 0.35 if mode == "train" else 0.20
    base_val    = _LABEL_BASE_VALENCE.get(label, 0.0) if label else 0.0

    cur_valence  = float(np.clip(base_val + rng.normal(0.0, noise_sigma), -1.0, 1.0))
    velocity     = float(np.clip(rng.normal(0.0, 0.25),                   -1.0, 1.0))

    _pos_base = max(0.0, base_val)
    _neg_base = max(0.0, -base_val)
    hist_pos = float(np.clip(_pos_base * 0.6 + rng.uniform(0.0, 0.3), 0.0, 1.0))
    hist_neu = float(np.clip(0.5 - abs(base_val) * 0.4 + rng.uniform(-0.1, 0.1), 0.0, 1.0))
    hist_neg = float(np.clip(_neg_base * 0.6 + rng.uniform(0.0, 0.3), 0.0, 1.0))

    if mode == "train" and rng.random() < 0.25:
        cur_valence = -cur_valence
        hist_pos, hist_neg = hist_neg, hist_pos

    is_strong   = label in _STRONG_LABELS if label else False
    conf_base   = float(rng.uniform(0.55, 0.85) if is_strong else rng.uniform(0.35, 0.65))
    alpha_raw   = rng.dirichlet([0.5] * N_CDM_STATES)
    alpha_raw[cdm_state] += conf_base * 6.0
    alpha_raw   /= alpha_raw.sum()
    hmm_conf    = float(alpha_raw.max())
    hmm_ent     = float(-np.sum(alpha_raw * np.log(alpha_raw + 1e-12)))
    hmm_emit    = float(rng.beta(3.0, 2.0) if is_strong else rng.beta(2.0, 3.0))
    top3_next   = sorted(rng.dirichlet([1.0] * 3).tolist(), reverse=True)
    intent_stab = float(rng.beta(3.0, 2.0) if label else rng.beta(1.0, 4.0))

    ctx = (
        cdm_one_hot            +
        [residency]            +
        transition             +
        [abruptness,
         coherence,
         entropy,
         spk_divergence,
         velocity,
         acceleration,
         hist_pos,
         hist_neu,
         hist_neg,
         resonance,
         volatility,
         cur_valence,
         msg_length,
         latency_norm,
         hmm_conf,
         hmm_ent,
         hmm_emit,
        ] + top3_next +
        [intent_stab]
        + [0.0] * PRIOR_DIM
        + [0.0] * (SARCASM_DIM + DYNAMICS_DIM + APPRAISAL_DIM)
    )

    if mode == "train":
        mask = (rng.random(CONTEXT_DIM) > 0.15).astype(float)
        ctx  = [float(v) * m for v, m in zip(ctx, mask)]

    assert len(ctx) == CONTEXT_DIM, f"ctx dim={len(ctx)} != {CONTEXT_DIM}"
    return ctx


def extract_csv_set_features(
    name: str, vader_analyzer, bert_analyzer, goe_analyzer, batch_size: int = 32,
) -> tuple:
    cfg   = CSV_SETS[name]
    tag   = cfg["tag"]
    path  = cfg["path"]
    empty = np.empty((0, FEATURE_DIM), dtype=np.float32)

    if not path.exists():
        if cfg["generated"]:
            logger.info(f"[{tag}] No sample file at {path} — skipping (run register_gen.py to create).")
        else:
            logger.warning(f"[{tag}] CSV not found at {path} — skipping.")
        return empty, [], []

    cache_path = _MODEL_DIR / f"{name}_cache.pkl"
    cache_key  = f"{cfg['version']}_{FEATURE_DIM}_{path.stat().st_mtime}"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                if cfg["gs"] == "goe":
                    X, y, gs = cached["data"]
                else:
                    X, y = cached["data"]
                    gs   = [{label: 1.0} for label in y]
                logger.info(f"[{tag}] Cache hit — {len(y)} samples, skipping NLP.")
                return np.array(X, dtype=np.float32), y, gs
            logger.info(f"[{tag}] Cache stale — recomputing.")
        except Exception as e:
            logger.warning(f"[{tag}] Cache load failed: {e} — recomputing.")

    pairs: list = []
    label_counts: Counter = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("goemotions_label", "").strip()
            text  = row.get("text", "").strip()
            if not text or label not in EMOTION_LABELS:
                continue
            cap = cfg["label_caps"].get(label)
            if cap is not None and label_counts[label] >= cap:
                continue
            pairs.append((text, label))
            label_counts[label] += 1
            if cfg["max_rows"] and len(pairs) >= cfg["max_rows"]:
                break

    if not pairs:
        logger.warning(f"[{tag}] No valid pairs in {path.name}.")
        return empty, [], []

    logger.info(f"[{tag}] {len(pairs)} samples ({len(label_counts)} classes). Running NLP...")
    texts, labels = [t for t, _ in pairs], [l for _, l in pairs]
    t0 = time.time()
    vader_outs, bert_outs, goe_outs = _run_parallel_batches(
        bert_analyzer, goe_analyzer, texts,
        vader_analyzer=vader_analyzer, batch_size=batch_size, label_prefix=tag,
    )
    logger.info(f"[{tag}] NLP done in {time.time()-t0:.1f}s")

    features = []
    for label, v, b, g in zip(labels, vader_outs, bert_outs, goe_outs):
        if cfg["ctx"] == "synthetic":
            ctx = build_synthetic_context_vector(label=label, mode="train")
            fv  = build_feature_vector(
                {"vader": v, "basic_bert": b, "go_emotions": g},
                context_vector=ctx[:CDM_CTX_DIM],
                trajectory_prior=ctx[CDM_CTX_DIM:CDM_CTX_DIM + PRIOR_DIM],
            )
        else:
            fv = build_feature_vector({"vader": v, "basic_bert": b, "go_emotions": g})
        features.append(fv.flatten())
    X = np.array(features, dtype=np.float32)

    if cfg["gs"] == "goe":
        gs = goe_outs
        cache_data = (X, labels, gs)
    else:
        gs = [{label: 1.0} for label in labels]
        cache_data = (X, labels)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"cache_key": cache_key, "data": cache_data}, f)
    except Exception as e:
        logger.warning(f"[{tag}] Cache write failed: {e}")

    return X, labels, gs
