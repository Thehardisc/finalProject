"""
trainer.py v2.0 — Periodic retraining for central_responder_service.

Key changes from v1:
  - build_synthetic_context_vector() generates label-correlated but heavily
    noisy context instead of all-zeros, preventing the zero-weight anomaly.
  - train_gating_network() trains GatingEnsembleNet (PyTorch) with:
      · label smoothing + class-balanced CrossEntropy
      · load-balance regularizer to prevent expert collapse
      · early stopping on val macro-F1
  - run_one_cycle() deploys a GatingNetworkWrapper (.pkl) with the same
    interface as the v1 sklearn Pipeline — no changes required in main.py.

Configuration (env vars, same as v1):
  RETRAIN_INTERVAL_SECONDS  (default: 1800)
  ACCURACY_GATE             (default: 0.40)
  MAX_EMPATHETIC_SAMPLES    (default: 25000)
  MIN_DB_SAMPLES            (default: 50)
"""

import os
import re
import gc
import sys
import json
import time
import pickle
import logging
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


# ── HuggingFace / stderr verbose capture ────────────────────────────────────────

def _setup_hf_logging():
    """Route HuggingFace datasets + transformers logging into our trainer logger."""
    _ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

    class _HFHandler(logging.Handler):
        def emit(self, record):
            msg = _ANSI.sub('', self.format(record)).strip()
            if msg:
                logger.info(f"[HF] {msg}")

    handler = _HFHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    for name in ("datasets", "datasets.utils", "datasets.builder",
                 "datasets.download", "fsspec", "transformers"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
        lg.propagate = False


_setup_hf_logging()


class _StderrToLogger:
    """Context manager: redirect stderr lines to logger.info during downloads."""
    _ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')

    def __init__(self):
        self._real = None
        self._buf  = ""

    def write(self, data):
        self._buf += data
        # tqdm uses \r to overwrite lines — treat both \r and \n as line ends
        for chunk in re.split(r'[\r\n]', self._buf):
            pass  # iterate to last segment
        lines = re.split(r'[\r\n]', self._buf)
        self._buf = lines[-1]  # keep incomplete last segment
        for line in lines[:-1]:
            line = self._ANSI.sub('', line).strip()
            if line:
                logger.info(f"[HF] {line}")

    def flush(self):
        pass

    def fileno(self):
        return self._real.fileno()

    def __enter__(self):
        self._real = sys.stderr
        sys.stderr  = self
        return self

    def __exit__(self, *_):
        sys.stderr = self._real

# ── Database ────────────────────────────────────────────────────────────────────
DB_USER     = os.getenv("POSTGRES_USER",     "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME     = os.getenv("POSTGRES_DB",       "emotion_db")
DB_HOST     = os.getenv("DB_HOST",           "db")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

# ── Configuration ───────────────────────────────────────────────────────────────
RETRAIN_INTERVAL        = int(os.environ.get("RETRAIN_INTERVAL_SECONDS",   1800))
ACCURACY_GATE           = float(os.environ.get("ACCURACY_GATE",            0.40))
MAX_EMPATHETIC_SAMPLES  = int(os.environ.get("MAX_EMPATHETIC_SAMPLES",     25_000))
MIN_DB_SAMPLES          = int(os.environ.get("MIN_DB_SAMPLES",             50))
MODEL_PATH              = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
META_PATH               = MODEL_PATH.with_name("meta_weights_meta.json")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from shared.constants import (
    EMOTION_LABELS, VADER_KEYS, BERT_LABELS,
    FEATURE_DIM, CONTEXT_DIM, CDM_CTX_DIM, PRIOR_DIM, ML_DIM, N_CDM_STATES,
    CTX_CDM_PROBS, CTX_RESIDENCY, CTX_TRANSITION, CTX_ABRUPTNESS,
    CTX_COHERENCE, CTX_ENTROPY, CTX_SPK_DIVERGENCE,
    CTX_VELOCITY, CTX_ACCELERATION,
    CTX_HIST_POS, CTX_HIST_NEU, CTX_HIST_NEG,
    CTX_RESONANCE, CTX_VOLATILITY, CTX_CURR_VALENCE,
    CTX_MSG_LENGTH, CTX_LATENCY_MS,
    CTX_HMM_CONF, CTX_HMM_ENTROPY, CTX_HMM_EMISSION, CTX_HMM_NEXT3,
    CTX_INTENT_STAB,
)
from meta_learner import build_feature_vector, GatingEnsembleNet, GatingNetworkWrapper, D_MODEL, DROPOUT

# ── Label → likely CDM intent state (primary, secondary, tertiary) ───────────────
# Used to generate label-correlated CDM state in synthetic context so the network
# can learn to use context features during training on GoEmotions.
_LABEL_TO_INTENT: dict = {
    'admiration':     (2,  14,  1),   # PRAISE, AGREEMENT, WARMTH
    'amusement':      (4,  10,  0),   # HUMOR, CURIOSITY, NEUTRAL
    'approval':       (14,  2, 11),   # AGREEMENT, PRAISE, ASSERTIVENESS
    'caring':         (12,  1,  9),   # EMPATHY, WARMTH, RECONCILIATION
    'curiosity':      (10,  3,  0),   # CURIOSITY, HELP_REQUEST, NEUTRAL
    'desire':         (1,  10,  0),   # WARMTH, CURIOSITY, NEUTRAL
    'excitement':     (2,  14,  4),   # PRAISE, AGREEMENT, HUMOR
    'gratitude':      (14,  1,  2),   # AGREEMENT, WARMTH, PRAISE
    'joy':            (1,   2, 14),   # WARMTH, PRAISE, AGREEMENT
    'love':           (1,  12,  9),   # WARMTH, EMPATHY, RECONCILIATION
    'optimism':       (14,  1,  2),   # AGREEMENT, WARMTH, PRAISE
    'pride':          (2,  11, 14),   # PRAISE, ASSERTIVENESS, AGREEMENT
    'relief':         (9,  14,  0),   # RECONCILIATION, AGREEMENT, NEUTRAL
    'realization':    (0,  10, 11),   # NEUTRAL, CURIOSITY, ASSERTIVENESS
    'anger':          (6,   5,  7),   # CONFLICT, TENSION, ARGUMENT
    'annoyance':      (5,  13,  6),   # TENSION, FRUSTRATION, CONFLICT
    'disapproval':    (5,  11,  6),   # TENSION, ASSERTIVENESS, CONFLICT
    'disgust':        (6,   5,  0),   # CONFLICT, TENSION, NEUTRAL
    'disappointment': (8,  13,  0),   # WITHDRAWAL, FRUSTRATION, NEUTRAL
    'embarrassment':  (8,   0, 13),   # WITHDRAWAL, NEUTRAL, FRUSTRATION
    'fear':           (3,   8,  0),   # HELP_REQUEST, WITHDRAWAL, NEUTRAL
    'grief':          (8,  12,  0),   # WITHDRAWAL, EMPATHY, NEUTRAL
    'nervousness':    (3,   8, 13),   # HELP_REQUEST, WITHDRAWAL, FRUSTRATION
    'remorse':        (9,   8, 12),   # RECONCILIATION, WITHDRAWAL, EMPATHY
    'sadness':        (8,  12,  0),   # WITHDRAWAL, EMPATHY, NEUTRAL
    'confusion':      (11,  3,  0),   # ASSERTIVENESS, HELP_REQUEST, NEUTRAL
    'neutral':        (0,  10, 11),   # NEUTRAL, CURIOSITY, ASSERTIVENESS
    'surprise':       (4,  10,  0),   # HUMOR, CURIOSITY, NEUTRAL
}

# Labels with strong, unambiguous emotional intent — higher hmm_confidence
_STRONG_LABELS = frozenset({
    'anger', 'joy', 'love', 'grief', 'admiration', 'fear',
    'disgust', 'gratitude', 'pride', 'remorse', 'sadness', 'excitement',
})

# ── Approximate valence baselines for synthetic context generation ──────────────
# These are deliberately imprecise (±0.10 vs _PREV_EMOTION_VALENCE in meta_learner.py)
# so the network cannot learn a lookup table mapping context → label.
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


# ── Synthetic context generation ────────────────────────────────────────────────

def build_synthetic_context_vector(
    label: str = None,
    mode: str = "train",
) -> list:
    """
    Generate a 23-dim synthetic context vector for static dataset samples.

    The critical invariant: valence-related features are label-correlated but
    NOT deterministically derived from the label.  Three mechanisms enforce this:

      1. Gaussian noise  σ=0.35 on all valence scalars — SNR ≈ 1.8:1 for
         strong emotions (e.g. joy base=0.82, noise keeps [-0.2, 1.8] → clipped)
      2. Adversarial flip (25% chance) — injects the wrong polarity to force
         the network to treat context as a soft prior, not a cheat sheet
      3. Per-feature dropout (15%) — simulates missing sensors / cold starts

    Modes:
      "train"  — full augmentation (items 1–3 above)
      "val"    — moderate noise only (σ=0.20, no adversarial flip)
      "cold"   — all zeros (for live SQL data with no conversation history)
    """
    if mode == "cold":
        return [0.0] * CONTEXT_DIM

    rng = np.random.RandomState()  # unseeded — each call is independent

    # ── CDM intent state: label-correlated when label is known ──────────────────
    if label and label in _LABEL_TO_INTENT:
        primary, secondary, tertiary = _LABEL_TO_INTENT[label]
        # 20% adversarial: random state to prevent lookup-table memorization
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

    # ── Temporal / structural scalars (independent of label) ─────────────────
    residency  = float(rng.beta(2.0, 5.0))
    transition = [float(rng.randint(0, N_CDM_STATES) / float(N_CDM_STATES)) for _ in range(3)]
    abruptness = float(rng.beta(1.0, 3.0))

    # ── Semantic scalars (independent of label) ───────────────────────────────
    coherence      = float(rng.beta(3.0, 2.0))
    entropy        = float(rng.beta(2.0, 3.0))
    spk_divergence = float(rng.beta(1.0, 4.0))
    acceleration   = float(np.clip(rng.normal(0.0, 0.12), -1.0, 1.0))
    resonance      = float(rng.beta(2.0, 2.0))
    volatility     = float(rng.beta(1.5, 3.0))
    msg_length     = float(rng.beta(2.0, 5.0))
    latency_norm   = float(rng.beta(1.0, 4.0))

    # ── Valence scalars: label-correlated but heavily noisy ───────────────────
    noise_sigma = 0.35 if mode == "train" else 0.20
    base_val    = _LABEL_BASE_VALENCE.get(label, 0.0) if label else 0.0

    cur_valence  = float(np.clip(base_val + rng.normal(0.0, noise_sigma), -1.0, 1.0))
    velocity     = float(np.clip(rng.normal(0.0, 0.25),                   -1.0, 1.0))

    # Episodic memory: 3-dim sentiment vector (pos, neu, neg) correlated with label
    _pos_base = max(0.0, base_val)
    _neg_base = max(0.0, -base_val)
    hist_pos = float(np.clip(_pos_base * 0.6 + rng.uniform(0.0, 0.3), 0.0, 1.0))
    hist_neu = float(np.clip(0.5 - abs(base_val) * 0.4 + rng.uniform(-0.1, 0.1), 0.0, 1.0))
    hist_neg = float(np.clip(_neg_base * 0.6 + rng.uniform(0.0, 0.3), 0.0, 1.0))

    if mode == "train" and rng.random() < 0.25:
        cur_valence = -cur_valence
        hist_pos, hist_neg = hist_neg, hist_pos

    # ── HMM-derived features: label-correlated ───────────────────────────────
    # Build α as a peaked distribution around cdm_state so hmm_conf and
    # hmm_entropy are consistent with the intent state we already chose.
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
        cdm_one_hot            +   # [0:15]  CDM one-hot (15 intent states)
        [residency]            +   # [15]
        transition             +   # [16:19]
        [abruptness,               # [19]
         coherence,                # [20]
         entropy,                  # [21]
         spk_divergence,           # [22]
         velocity,                 # [23]
         acceleration,             # [24]
         hist_pos,                 # [25]
         hist_neu,                 # [26]
         hist_neg,                 # [27]
         resonance,                # [28]
         volatility,               # [29]
         cur_valence,              # [30]
         msg_length,               # [31]
         latency_norm,             # [32]
         hmm_conf,                 # [33]
         hmm_ent,                  # [34]
         hmm_emit,                 # [35]
        ] + top3_next +            # [36:39]
        [intent_stab]              # [39]
        + [0.0] * PRIOR_DIM        # [40:68] trajectory prior — zeros for static single-turn data
    )

    # Per-feature dropout: 15% of features zeroed on each training sample
    if mode == "train":
        mask = (rng.random(CONTEXT_DIM) > 0.15).astype(float)
        ctx  = [float(v) * m for v, m in zip(ctx, mask)]

    assert len(ctx) == CONTEXT_DIM, f"ctx dim={len(ctx)} != {CONTEXT_DIM}"
    return ctx


# ── Context encoder pre-training ───────────────────────────────────────────────

def pretrain_context_encoder(
    model: GatingEnsembleNet,
    X_ctx: np.ndarray,
    y_labels: list,
    classes: list,
    has_cdm: np.ndarray = None,
    n_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> GatingEnsembleNet:
    """
    Pre-train enc_ctx_expert + ctx_prior_head on context → emotion before
    full pipeline training. Forces the context encoder to learn emotion-predictive
    representations rather than noise, giving the Bayesian prior a useful starting point.

    Only trains enc_ctx_expert and ctx_prior_head — all other parameters frozen.
    When has_cdm is provided, only real-CDM samples are used for pretraining.
    """
    # Filter to real-CDM samples when mask is available
    if has_cdm is not None and has_cdm.any():
        X_ctx   = X_ctx[has_cdm]
        y_labels = [y_labels[i] for i, v in enumerate(has_cdm) if v]
        logger.info(f"  [CtxPretrain] Using {len(X_ctx)} real-CDM samples for pretraining.")
    elif has_cdm is not None and not has_cdm.any():
        logger.info("  [CtxPretrain] No real-CDM samples — skipping pretrain.")
        return model

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx.get(y, 0) for y in y_labels])

    X_ctx_t = torch.tensor(X_ctx[:, ML_DIM:].astype(np.float32), device=device)
    y_t     = torch.tensor(y_idx, dtype=torch.long, device=device)

    params  = list(model.enc_ctx_expert.parameters()) + \
              list(model.ctx_prior_head.parameters())
    opt     = torch.optim.Adam(params, lr=lr)

    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        logits = model.ctx_prior_head(model.enc_ctx_expert(X_ctx_t))
        loss   = F.cross_entropy(logits, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            logger.info(f"  [CtxPretrain] epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}")

    logger.info("  [CtxPretrain] Done — enc_ctx_expert primed.")
    model.eval()
    return model


# ── PyTorch training loop ───────────────────────────────────────────────────────

def train_gating_network(
    X_tr: np.ndarray,
    y_tr: list,
    X_v: np.ndarray,
    y_v: list,
    classes: list,
    has_cdm: np.ndarray = None,
    n_epochs: int = 80,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    load_balance_coeff: float = 5e-4,  # reduced: Bayesian arch self-regulates ctx weight
    patience: int = 10,
    device: str = None,
) -> GatingNetworkWrapper:
    """
    Train GatingEnsembleNet and return a sklearn-compatible GatingNetworkWrapper.

    Training strategy:
      - StandardScaler fitted on X_tr only (no leakage into val/test)
      - CrossEntropy with label_smoothing=0.10 + inverse-frequency class weights
      - Load-balance regularizer: λ·‖ᾱ − 1/3‖² prevents expert collapse
        (without it, GoEmotions tends to capture ~90% of gate weight)
      - OneCycleLR: fast convergence on small datasets
      - Early stopping on val macro-F1 (better metric than accuracy for 28 imbalanced classes)
      - Gradient clipping at 1.0 for training stability
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  [Trainer] Training on device={device}")

    # ── CDM masking: zero out CDM block for samples without real context ──────
    # When CDM[39:79] is synthetic (has_cdm=False), hmm_conf becomes 0
    # → ctx_weight ≈ 0.12 (minimum) → ctx_prior_head barely contributes.
    # This prevents the context encoder from learning spurious synthetic→label
    # correlations; only real-CDM samples (EmpatheticDialogues, collect.py)
    # train the context path meaningfully.
    CDM_SLICE = slice(ML_DIM, ML_DIM + CDM_CTX_DIM)
    if has_cdm is not None:
        no_cdm = ~has_cdm
        if no_cdm.any():
            X_tr = X_tr.copy()
            X_tr[no_cdm, CDM_SLICE] = 0.0
            logger.info(
                f"  [CDMMask] Zeroed CDM[{ML_DIM}:{ML_DIM+CDM_CTX_DIM}] for "
                f"{no_cdm.sum()}/{len(X_tr)} samples "
                f"({100*no_cdm.mean():.1f}% without real CDM)"
            )
        # Val/test data comes from GoEmotions (no real CDM).
        # Zero val CDM too so the scaler sees a consistent distribution
        # for CDM columns — prevents scaled val outliers when scaler is
        # fit on zeroed training CDM.
        X_v = X_v.copy()
        X_v[:, CDM_SLICE] = 0.0

    # ── Scaling ──────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_v_s   = scaler.transform(X_v)

    # ── Label encoding ────────────────────────────────────────────────────────
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_tr_idx = np.array([class_to_idx[y]          for y in y_tr])
    y_v_idx  = np.array([class_to_idx.get(y, 0)   for y in y_v])

    # ── Class weights: inverse frequency, capped at [0.1, 10] ────────────────
    counts   = np.bincount(y_tr_idx, minlength=len(classes)).clip(1)
    weights  = (1.0 / counts)
    weights  = (weights / weights.mean()).clip(0.1, 10.0)
    cw_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    # ── DataLoader ────────────────────────────────────────────────────────────
    ds     = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr_idx, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model = GatingEnsembleNet(n_classes=len(classes), d=D_MODEL, dropout=DROPOUT).to(device)

    # Pre-train context encoder — only on real-CDM samples when mask is available
    logger.info("  [CtxPretrain] Pre-training context encoder on context block...")
    model = pretrain_context_encoder(
        model, X_tr, y_tr, classes,
        has_cdm=has_cdm, n_epochs=20, device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(loader),
        epochs=n_epochs,
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.10)

    # ── NLP anchor: GoEmotions index → class_to_idx mapping ─────────────────
    # EMOTION_LABELS order ≠ class_to_idx order (classes may be a sorted subset).
    # Precompute a lookup so the anchor loss can convert goe_argmax → model class idx.
    goe_to_class_idx = torch.tensor(
        [class_to_idx.get(lbl, -1) for lbl in EMOTION_LABELS],
        dtype=torch.long, device=device,
    )  # [28]  — -1 for labels absent from training classes

    nlp_anchor_coeff  = 0.30   # weight of NLP anchor vs main CE loss
    nlp_anchor_thresh = 0.75   # GoEmotions confidence threshold to apply anchor

    uniform_gate  = torch.ones(4, device=device) / 4.0
    X_v_t         = torch.tensor(X_v_s, dtype=torch.float32, device=device)
    best_f1       = -1.0
    best_state    = None
    no_improve    = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits, alpha = model(xb)
            ce_loss    = criterion(logits, yb)

            # Load-balance: penalise deviation of mean gate weights from uniform.
            # Prevents GoEmotions from monopolising α without context signal.
            alpha_mean   = alpha.mean(dim=0)                         # [4]
            balance_loss = ((alpha_mean - uniform_gate) ** 2).sum()

            # NLP anchor: when GoEmotions is highly confident, steer logits toward
            # its top prediction. Teaches the model to trust NLP on clear-cut cases
            # rather than letting context override a 90%+ signal.
            goe_raw            = xb[:, 11:ML_DIM]                            # [B, 28]
            goe_conf, goe_ridx = goe_raw.max(dim=1)                          # [B]
            goe_cidx           = goe_to_class_idx[goe_ridx]                  # [B] class indices
            anchor_mask        = (goe_conf > nlp_anchor_thresh) & (goe_cidx >= 0)
            anchor_loss = (
                F.cross_entropy(logits[anchor_mask], goe_cidx[anchor_mask])
                if anchor_mask.any() else torch.tensor(0.0, device=device)
            )

            loss = ce_loss + load_balance_coeff * balance_loss + nlp_anchor_coeff * anchor_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        # ── Validation every epoch ───────────────────────────────────────────
        if True:
            model.eval()
            with torch.no_grad():
                logits_v, alpha_v = model(X_v_t)
                preds_v = logits_v.argmax(dim=-1).cpu().numpy()

            val_f1      = f1_score(y_v_idx, preds_v, average='macro', zero_division=0)
            alpha_mean_ = alpha_v.mean(dim=0).cpu().numpy()
            avg_loss    = epoch_loss / len(loader)
            ctx_str = f" ctx:{alpha_mean_[3]:.3f}]" if len(alpha_mean_) > 3 else "]"
            logger.info(
                f"  [Epoch {epoch+1:3d}/{n_epochs}]  "
                f"loss={avg_loss:.4f}  val_F1={val_f1:.4f}  "
                f"α=[vader:{alpha_mean_[0]:.3f} bert:{alpha_mean_[1]:.3f} goe:{alpha_mean_[2]:.3f}{ctx_str}"
            )

            if val_f1 > best_f1:
                best_f1    = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"  [EarlyStopping] No improvement for {patience} checks.")
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info(f"  [Trainer] Best checkpoint restored (val F1={best_f1:.4f})")

    model.eval()
    return GatingNetworkWrapper(model=model, classes=classes, scaler=scaler)


# ── EmpatheticDialogues conversation-context training data ─────────────────────

# Map EmpatheticDialogues 32 emotions → GoEmotions 28 labels
_EMPATHETIC_TO_GOEMOTION: dict = {
    'sentimental': 'love',       'afraid':       'fear',
    'proud':       'pride',      'faithful':     'caring',
    'terrified':   'fear',       'joyful':       'joy',
    'angry':       'anger',      'sad':          'sadness',
    'jealous':     'disapproval','grateful':     'gratitude',
    'embarrassed':  'embarrassment',
    'excited':     'excitement', 'annoyed':      'annoyance',
    'lonely':      'sadness',    'surprised':    'surprise',
    'furious':     'anger',      'disappointed': 'disappointment',
    'caring':      'caring',     'trusting':     'approval',
    'disgusted':   'disgust',    'anticipating': 'desire',          # anticipating→desire (wanting something to happen)
    'anxious':     'nervousness','nostalgic':    'curiosity',       # nostalgic→curiosity (wondering about the past)
    'confident':   'pride',
    'devastated':  'grief',      'hopeful':      'optimism',
    'guilty':      'remorse',    'impressed':    'admiration',
    'apprehensive':'nervousness','touched':      'caring',
}

# Rough valence for EmpatheticDialogues emotions (used to approximate velocity)
_EMPATHETIC_VALENCE: dict = {
    'joyful': 0.82, 'excited': 0.78, 'grateful': 0.72, 'proud': 0.68,
    'content': 0.55, 'hopeful': 0.60, 'anticipating': 0.40, 'trusting': 0.50,
    'caring': 0.58, 'faithful': 0.50, 'sentimental': 0.40, 'impressed': 0.65,
    'touched': 0.55, 'prepared': 0.35, 'nostalgic': 0.10, 'confident': 0.60,
    'surprised': 0.15,
    'angry': -0.72, 'furious': -0.85, 'sad': -0.68, 'terrified': -0.80,
    'afraid': -0.62, 'lonely': -0.65, 'disappointed': -0.58, 'annoyed': -0.48,
    'embarrassed': -0.42, 'disgusted': -0.70, 'devastated': -0.88,
    'jealous': -0.52, 'guilty': -0.58, 'anxious': -0.45, 'apprehensive': -0.40,
}


def _load_hmm_params():
    """Load HMM transmat and emissionprob from models/cdm_hmm.pkl."""
    hmm_path = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")).parent / "cdm_hmm.pkl"
    if not hmm_path.exists():
        return None, None
    try:
        import pickle as pkl
        with open(hmm_path, "rb") as f:
            d = pkl.load(f)
        return np.array(d["transmat"]), np.array(d["emissionprob"])
    except Exception as e:
        logger.warning(f"Could not load CDM HMM for EmpatheticDialogues: {e}")
        return None, None


def _hmm_forward_step(alpha, transmat, emissionprob, obs):
    """Single HMM forward step: α_t = normalise((α_{t-1} @ A) ⊙ B[:,obs])."""
    pred = alpha @ transmat
    upd  = pred * emissionprob[:, obs]
    s    = upd.sum()
    return upd / s if s > 1e-12 else np.ones(len(alpha)) / len(alpha)


def _empathetic_obs(emotion_label: str) -> int:
    """Map an EmpatheticDialogues emotion to a DailyDialog-compatible obs index."""
    # act=3 (commissive) for positive, act=1 (inform) for neutral/negative
    # emotion: map to 0-6 (neutral/anger/disgust/fear/happiness/sadness/surprise)
    _EMO_TO_7 = {
        'joy': 4, 'joyful': 4, 'excited': 4, 'grateful': 4, 'proud': 4,
        'content': 4, 'hopeful': 4, 'trusting': 4, 'caring': 4, 'impressed': 4,
        'anger': 1, 'angry': 1, 'furious': 1, 'annoyed': 1, 'jealous': 1,
        'disgust': 2, 'disgusted': 2, 'guilty': 2,
        'fear': 3, 'afraid': 3, 'terrified': 3, 'anxious': 3, 'apprehensive': 3,
        'sadness': 5, 'sad': 5, 'lonely': 5, 'devastated': 5, 'disappointed': 5,
        'embarrassed': 5, 'sentimental': 5,
        'surprise': 6, 'surprised': 6,
    }
    emot_idx = _EMO_TO_7.get(emotion_label.lower(), 0)
    val = _EMPATHETIC_VALENCE.get(emotion_label.lower(), 0.0)
    act_idx  = 3 if val > 0.3 else (1 if val > -0.3 else 2)
    return act_idx * 7 + emot_idx


def extract_empathetic_dialogues_features(
    vader_analyzer, bert_analyzer, goe_analyzer,
    split: str = "train",
) -> tuple:
    """
    Build (X [N, FEATURE_DIM], y [N], gs [N]) from EmpatheticDialogues.

    The dataset's native splits are used directly (train ~19k / val ~2.7k / test ~2.5k).
    Size is capped by MAX_EMPATHETIC_SAMPLES (train) and MAX_EMPATHETIC_SAMPLES // 5 (val/test).

    Labels: 32 EmpatheticDialogues emotions → GoEmotions 28 via _EMPATHETIC_TO_GOEMOTION.
    CDM context: synthetic (has_cdm=False — single-turn data, no conversation history).
    ctx_mode: "train" for train split, "val" for val, "cold" for test.
    """
    hf_split = {"train": "train", "val": "validation", "test": "test"}.get(split, "train")
    cap      = MAX_EMPATHETIC_SAMPLES if split == "train" else MAX_EMPATHETIC_SAMPLES // 5
    ctx_mode = {"train": "train", "val": "val", "test": "cold"}[split]

    try:
        from datasets import load_dataset
        emp = load_dataset("bdotloh/empathetic-dialogues-contexts", split=hf_split)
    except Exception as e:
        logger.warning(f"EmpatheticDialogues load failed: {e} — skipping {split} split.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), [], []

    rows = list(emp)[:cap]

    total    = len(rows)
    mapped   = sum(1 for r in rows if _EMPATHETIC_TO_GOEMOTION.get(str(r.get('emotion', '')).lower()) is not None)
    skipped  = total - mapped
    logger.info(
        f"  [EmpDialogues] {split}: {total} rows loaded, "
        f"{mapped} mapped ({skipped} skipped — unmapped emotions: "
        f"prepared/content/ashamed)"
    )
    t0 = time.time()

    features, labels, gs_list = [], [], []
    for i, row in enumerate(rows):
        if i > 0 and i % 500 == 0:
            elapsed  = time.time() - t0
            rate     = i / elapsed
            eta_s    = int((total - i) / rate) if rate > 0 else 0
            logger.info(
                f"  [EmpDialogues] {split} {i}/{total} rows "
                f"({len(features)} kept so far, {rate:.1f} rows/s, ETA {eta_s}s)"
            )

        text        = str(row.get('situation', '')).strip()
        raw_emotion = str(row.get('emotion', 'neutral')).lower()
        goemo_label = _EMPATHETIC_TO_GOEMOTION.get(raw_emotion)

        if not text or goemo_label is None:
            continue
        try:
            vader_out = vader_analyzer(text) if callable(vader_analyzer) else {}
            bert_out  = bert_analyzer(text)  if callable(bert_analyzer)  else {}
            goe_out   = goe_analyzer(text)   if callable(goe_analyzer)   else {}
        except Exception:
            continue

        ctx = build_synthetic_context_vector(label=goemo_label, mode=ctx_mode)
        fv  = build_feature_vector(
            {"vader": vader_out, "basic_bert": bert_out, "go_emotions": goe_out},
            context_vector=ctx[:CDM_CTX_DIM],
            trajectory_prior=ctx[CDM_CTX_DIM:],
        )
        features.append(fv.flatten())
        labels.append(goemo_label)
        gs_list.append(goe_out)

    elapsed = time.time() - t0
    logger.info(
        f"  [EmpDialogues] Extracted {len(features)} samples ({split}) "
        f"in {elapsed:.0f}s ({len(features)/elapsed:.1f} samples/s)."
    )
    empty = np.empty((0, FEATURE_DIM), dtype=np.float32)
    return (
        np.array(features, dtype=np.float32) if features else empty,
        labels,
        gs_list,
    )


# ── Synthetic sentence generation for missing GoEmotions classes ──────────────

# Per-class target counts — exactly replace what was removed from EmpatheticDialogues:
#   neutral:     prepared(584) + ashamed(485 fallback) = 1069 removed
#   relief:      content(568) = 568 removed
#   amusement / confusion / realization: never in ED → use 568 as baseline (matches relief)
_SYNTHETIC_COUNTS: dict = {
    "neutral":     1069,
    "relief":       568,
    "amusement":    568,
    "confusion":    568,
    "realization":  568,
}
_SYNTHETIC_CLASSES = list(_SYNTHETIC_COUNTS.keys())
_BATCH_SIZE        = 250   # max sentences per Claude API call (fits in 4096 tokens)


def _generate_synthetic_sentences() -> dict:
    """
    Call Claude Haiku to generate synthetic training sentences for each missing class.
    Loops in batches of _BATCH_SIZE until the per-class target in _SYNTHETIC_COUNTS is met.
    """
    import anthropic
    client    = anthropic.Anthropic()
    result    = {}
    total_all = sum(_SYNTHETIC_COUNTS.values())
    done_all  = 0

    for label, target in _SYNTHETIC_COUNTS.items():
        sentences: list = []
        n_batches_est = -(-target // _BATCH_SIZE)  # estimate assuming full yield per call
        logger.info(
            f"  [Synthetic] '{label}': target={target} sentences "
            f"(~{n_batches_est} batches of ≤{_BATCH_SIZE}, may need more if API returns fewer)"
        )
        batch_num = 0
        while len(sentences) < target:
            batch_num += 1
            needed = min(_BATCH_SIZE, target - len(sentences))
            logger.info(
                f"  [Synthetic]   batch {batch_num} for '{label}' "
                f"— requesting {needed}, have {len(sentences)}/{target} so far..."
            )
            prompt = (
                f"Generate {needed} diverse first-person situational sentences "
                f"expressing the emotion '{label}'. "
                "Style: EmpatheticDialogues 'situation' field — 1-2 sentences describing "
                "a real-life context where someone feels this emotion. "
                "Vary the settings: work, relationships, hobbies, discovery, everyday moments. "
                "One situation per line. No numbering, no bullets."
            )
            msg   = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            batch = [l.strip() for l in msg.content[0].text.splitlines() if l.strip()]
            sentences.extend(batch)
            logger.info(
                f"  [Synthetic]   batch {batch_num} done — "
                f"got {len(batch)}, total so far: {len(sentences)}/{target}"
            )
        result[label] = sentences[:target]
        done_all     += len(result[label])
        logger.info(
            f"  [Synthetic] '{label}' complete: {len(result[label])}/{target} sentences "
            f"[overall {done_all}/{total_all}]"
        )
    return result


_GOE_DIRECT_PER_CLASS   = 500    # samples per class from GoEmotions train+val+test
_GOE_DIRECT_CACHE_ID    = "goemotions_direct_v3"


def extract_goemotions_direct_features(
    vader_analyzer, bert_analyzer, goe_analyzer,
) -> tuple:
    """
    Load GoEmotions val+test samples and build NLP-aligned training features.

    Unlike EmpatheticDialogues (where label = conversation topic), these samples
    have gold labels that *match* what the GoEmotions model outputs. Training on
    them teaches the meta-learner to trust NLP signals when they are confident.

    Uses val+test splits (not train) so the GoEmotions model's output is realistic
    rather than over-fitted.  Target: _GOE_DIRECT_PER_CLASS samples per class,
    balanced across all 28 labels.

    Results are cached in MODEL_PATH.parent/goemotions_direct_cache.pkl.
    """
    import hashlib

    cache_path = MODEL_PATH.parent / "goemotions_direct_cache.pkl"
    cache_key  = f"{_GOE_DIRECT_CACHE_ID}_{_GOE_DIRECT_PER_CLASS}_{FEATURE_DIM}"
    empty      = np.empty((0, FEATURE_DIM), dtype=np.float32)

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                X, y, gs = cached["data"]
                logger.info(
                    f"[GoEDirect] Feature cache hit — loaded {len(y)} NLP-aligned samples."
                )
                return np.array(X, dtype=np.float32) if X else empty, y, gs
            logger.info("[GoEDirect] Cache stale — recomputing.")
        except Exception as e:
            logger.warning(f"[GoEDirect] Cache load failed: {e} — recomputing.")

    logger.info(
        f"[GoEDirect] Loading GoEmotions train+val+test (≤{_GOE_DIRECT_PER_CLASS}/class × 28 classes)..."
    )
    try:
        from datasets import load_dataset
        with _StderrToLogger():
            logger.info("[GoEDirect] Downloading split=train...")
            ds_train = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
            logger.info(f"[GoEDirect] train loaded ({len(ds_train)} rows). Downloading split=validation...")
            ds_val   = load_dataset("google-research-datasets/go_emotions", "simplified", split="validation")
            logger.info(f"[GoEDirect] validation loaded ({len(ds_val)} rows). Downloading split=test...")
            ds_test  = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
            logger.info(f"[GoEDirect] test loaded ({len(ds_test)} rows).")
    except Exception as e:
        logger.warning(f"[GoEDirect] Could not load GoEmotions dataset: {e} — skipping.")
        return empty, [], []

    # Dataset label names (28 GoEmotions classes, may differ in order from EMOTION_LABELS)
    id2label = ds_val.features["labels"].feature.int2str

    # Collect balanced samples per class — train first (most examples), then val+test
    per_class: dict = {lbl: [] for lbl in EMOTION_LABELS}
    for row in list(ds_train) + list(ds_val) + list(ds_test):
        if len(row["labels"]) != 1:
            continue   # skip multi-label rows
        label_str = id2label(row["labels"][0])
        if label_str in per_class and len(per_class[label_str]) < _GOE_DIRECT_PER_CLASS:
            per_class[label_str].append(row["text"])

    counts = {k: len(v) for k, v in per_class.items() if v}
    total  = sum(counts.values())
    logger.info(f"[GoEDirect] Collected {total} samples across {len(counts)} classes: {counts}")

    if total == 0:
        return empty, [], []

    # Build flat (text, label) pairs
    pairs = [(text, lbl) for lbl, texts in per_class.items() for text in texts]

    # Batched NLP inference
    all_texts = [t for t, _ in pairs]
    logger.info(f"[GoEDirect] Running batched NLP on {len(all_texts)} texts...")
    t0 = time.time()

    vader_outs = []
    for text in all_texts:
        try:
            vader_outs.append(vader_analyzer(text) if callable(vader_analyzer) else {})
        except Exception:
            vader_outs.append({})

    bert_outs = _run_batch(bert_analyzer, all_texts, label="GoEDirect/BERT") if callable(bert_analyzer) else [{} for _ in all_texts]
    goe_outs  = _run_batch(goe_analyzer,  all_texts, label="GoEDirect/GoE")  if callable(goe_analyzer)  else [{} for _ in all_texts]

    elapsed = time.time() - t0
    logger.info(
        f"[GoEDirect] NLP done — {len(all_texts)} texts in {elapsed:.0f}s "
        f"({len(all_texts)/elapsed:.1f} samples/s)"
    )

    # Build feature vectors (no CDM context — zero by default)
    features, labels, gs_list = [], [], []
    for (_, lbl), vader_out, bert_out, goe_out in zip(pairs, vader_outs, bert_outs, goe_outs):
        fv = build_feature_vector(
            {"vader": vader_out, "basic_bert": bert_out, "go_emotions": goe_out},
        )
        features.append(fv.flatten())
        labels.append(lbl)
        gs_list.append(goe_out)

    logger.info(f"[GoEDirect] Built {len(features)} NLP-aligned feature vectors.")

    # Cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"cache_key": cache_key, "data": (features, labels, gs_list)}, f)
        logger.info(f"[GoEDirect] Cache saved → {cache_path}")
    except Exception as e:
        logger.warning(f"[GoEDirect] Could not save cache: {e}")

    return np.array(features, dtype=np.float32) if features else empty, labels, gs_list


def load_synthetic_features(vader_analyzer, bert_analyzer, goe_analyzer) -> tuple:
    """
    Load (or auto-generate) synthetic sentences for the 5 missing GoEmotions classes
    and process them through the same NLP pipeline as EmpatheticDialogues.

    Sentences are stored in MODEL_PATH.parent/synthetic_sentences.json (bind-mounted,
    persists container rebuilds). Computed features are cached in
    synthetic_features_cache.pkl keyed by the JSON file's MD5 hash — NLP only runs
    once per unique JSON file.
    """
    import hashlib

    json_path   = MODEL_PATH.parent / "synthetic_sentences.json"
    cache_path  = MODEL_PATH.parent / "synthetic_features_cache.pkl"
    empty       = np.empty((0, FEATURE_DIM), dtype=np.float32)

    # ── 1. Obtain JSON (generate if missing) ────────────────────────────────────
    if not json_path.exists() or json_path.stat().st_size == 0:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "[Synthetic] ANTHROPIC_API_KEY is not set — cannot generate synthetic sentences. "
                "Add ANTHROPIC_API_KEY to .env and rebuild to cover: "
                f"{_SYNTHETIC_CLASSES}. Training will proceed with 23/28 classes."
            )
            return empty, [], []

        total_needed = sum(_SYNTHETIC_COUNTS.values())
        logger.info(
            f"[Synthetic] synthetic_sentences.json not found — generating {total_needed} sentences "
            f"across {len(_SYNTHETIC_CLASSES)} classes via Claude Haiku: {_SYNTHETIC_COUNTS}"
        )
        try:
            data = _generate_synthetic_sentences()
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            saved = sum(len(v) for v in data.values())
            logger.info(f"[Synthetic] Saved {saved} sentences → {json_path}")
        except Exception as e:
            logger.warning(f"[Synthetic] Generation failed: {e} — training with 23/28 classes.")
            return empty, [], []
    else:
        with open(json_path) as f:
            data = json.load(f)
        loaded = {k: len(v) for k, v in data.items()}
        logger.info(f"[Synthetic] Loaded existing synthetic_sentences.json: {loaded}")

    # ── 2. Compute syn_hash to check feature cache ──────────────────────────────
    syn_hash = hashlib.md5(json_path.read_bytes()).hexdigest()[:12]

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("syn_hash") == syn_hash and cached.get("feature_dim") == FEATURE_DIM:
                X_syn, y_syn, gs_syn = cached["data"]
                logger.info(
                    f"[Synthetic] Feature cache hit (hash={syn_hash}) — "
                    f"loaded {len(y_syn)} pre-computed feature vectors. Skipping NLP."
                )
                return (
                    np.array(X_syn, dtype=np.float32) if X_syn else empty,
                    y_syn,
                    gs_syn,
                )
            else:
                logger.info("[Synthetic] Feature cache stale (hash or dim mismatch) — recomputing.")
        except Exception as e:
            logger.warning(f"[Synthetic] Cache load failed: {e} — recomputing.")

    # ── 3. Batch NLP processing ─────────────────────────────────────────────────
    # Collect all (label, text) pairs; keep only valid GoEmotions labels
    ordered_pairs: list = []  # (goemo_label, text)
    for goemo_label, sentences in data.items():
        if goemo_label not in EMOTION_LABELS:
            logger.warning(f"[Synthetic] Unknown label '{goemo_label}' in JSON — skipping.")
            continue
        for text in sentences:
            if text:
                ordered_pairs.append((goemo_label, text))

    total_sentences = len(ordered_pairs)
    logger.info(
        f"[Synthetic] Running batched NLP on {total_sentences} sentences "
        f"(batch_size=32, 3 models)..."
    )
    t0 = time.time()

    all_texts = [t for _, t in ordered_pairs]

    # VADER is pure regex — fast even per-sentence, but we keep it consistent
    vader_results = []
    for text in all_texts:
        try:
            vader_results.append(vader_analyzer(text) if callable(vader_analyzer) else {})
        except Exception:
            vader_results.append({})

    # BERT and GoEmotions: batched
    bert_results = _run_batch(bert_analyzer, all_texts, label="Synthetic/BERT") if callable(bert_analyzer) else [{} for _ in all_texts]
    goe_results  = _run_batch(goe_analyzer,  all_texts, label="Synthetic/GoE")  if callable(goe_analyzer)  else [{} for _ in all_texts]

    elapsed_nlp = time.time() - t0
    rate = total_sentences / elapsed_nlp if elapsed_nlp > 0 else 0
    logger.info(
        f"[Synthetic] Batched NLP done — {total_sentences} sentences in {elapsed_nlp:.0f}s "
        f"({rate:.1f} samples/s)"
    )

    # ── 4. Build feature vectors ─────────────────────────────────────────────────
    features, labels, gs_list = [], [], []
    class_counts: dict = {}
    for (goemo_label, _text), vader_out, bert_out, goe_out in zip(
        ordered_pairs, vader_results, bert_results, goe_results
    ):
        ctx = build_synthetic_context_vector(label=goemo_label, mode="train")
        fv  = build_feature_vector(
            {"vader": vader_out, "basic_bert": bert_out, "go_emotions": goe_out},
            context_vector=ctx[:CDM_CTX_DIM],
            trajectory_prior=ctx[CDM_CTX_DIM:],
        )
        features.append(fv.flatten())
        labels.append(goemo_label)
        gs_list.append(goe_out)
        class_counts[goemo_label] = class_counts.get(goemo_label, 0) + 1

    for lbl, cnt in class_counts.items():
        logger.info(f"  [Synthetic] '{lbl}': {cnt} feature vectors built")

    # ── 5. Save feature cache ────────────────────────────────────────────────────
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "syn_hash":    syn_hash,
                "feature_dim": FEATURE_DIM,
                "data":        (features, labels, gs_list),
            }, f)
        logger.info(f"[Synthetic] Feature cache saved → {cache_path} (hash={syn_hash})")
    except Exception as e:
        logger.warning(f"[Synthetic] Could not save feature cache: {e}")

    return (
        np.array(features, dtype=np.float32) if features else empty,
        labels,
        gs_list,
    )


# ── Relabeled conversations (implicit emotion labels from Claude API) ──────────

RELABELED_DATA_PATH = Path(os.environ.get(
    "RELABELED_DATA_PATH",
    "/app/training_data/conversations_relabeled.jsonl",
))


def load_relabeled_conversations() -> tuple:
    """
    Load re-labeled conversations produced by relabel.py.

    Key difference from GoEmotions training data:
      - Labels come from Claude's implicit emotion recognition (not GoEmotions predictions)
      - NLP features are the real pipeline outputs stored in conversations.jsonl
      - GoEmotions features in the vector may DISAGREE with the label →
        the model learns that GoEmotions can be wrong and context matters

    CDM context is synthetic (correlated with the Claude-assigned label, same
    augmentation as GoEmotions training) — real CDM vectors are not stored in
    the collected data at sufficient resolution to reconstruct the full 40-dim block.
    """
    if not RELABELED_DATA_PATH.exists():
        logger.info(f"  [Relabeled] {RELABELED_DATA_PATH} not found — skipping.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    try:
        conversations = [
            json.loads(line)
            for line in RELABELED_DATA_PATH.read_text().splitlines()
            if line.strip()
        ]
    except Exception as e:
        logger.warning(f"  [Relabeled] Failed to read file: {e} — skipping.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    features, labels = [], []

    for conv in conversations:
        chunks   = conv.get("relabeled_chunks", [])
        messages = conv.get("messages", [])

        for chunk in chunks:
            emotions_dict = chunk.get("emotions", {})
            if not emotions_dict:
                continue

            # Dominant emotion = argmax of Claude's 28-dim scores
            valid = {k: v for k, v in emotions_dict.items() if k in EMOTION_LABELS}
            if not valid:
                continue
            chunk_label = max(valid, key=valid.get)

            for idx in chunk.get("message_indices", []):
                if idx >= len(messages):
                    continue
                stages = messages[idx].get("pipeline", {}).get("stages", {})
                if not stages:
                    continue

                # Map stored stage keys → build_feature_vector format
                model_outputs = {
                    "vader":       stages.get("vader",       {}),
                    "basic_bert":  stages.get("bert",        {}),
                    "go_emotions": stages.get("goemotions",  {}),
                }

                ctx = build_synthetic_context_vector(label=chunk_label, mode="train")
                fv  = build_feature_vector(
                    model_outputs,
                    context_vector=ctx[:CDM_CTX_DIM],
                    trajectory_prior=ctx[CDM_CTX_DIM:],
                )
                features.append(fv.flatten())
                labels.append(chunk_label)

    logger.info(
        f"  [Relabeled] Loaded {len(features)} samples "
        f"from {len(conversations)} conversations."
    )
    return (
        np.array(features, dtype=np.float32) if features
        else np.empty((0, FEATURE_DIM), dtype=np.float32),
        labels,
    )


# ── MELD conversation-context training data ────────────────────────────────────

def _download_meld_raw(out_path: Path, max_rows: int = 3000) -> None:
    """Download MELD and save only (dialogue_id, utterance_id, text, emotion, speaker)
    to a compact JSON. Called BEFORE NLP models are loaded so HuggingFace dataset and
    BERT/GoEmotions don't share RAM simultaneously."""
    try:
        from datasets import load_dataset
        slice_size = min(max_rows * 2, 4000)
        logger.info(f"[MELD] Pre-downloading raw data (train[:{slice_size}]) — no NLP models in memory yet...")
        ds = load_dataset("declare-lab/MELD", split=f"train[:{slice_size}]")
        rows = []
        for row in ds:
            rows.append({
                "d": str(row.get("Dialogue_ID", "")),
                "u": int(row.get("Utterance_ID", 0)),
                "t": str(row.get("Utterance", "")).strip(),
                "e": str(row.get("Emotion",   "neutral")).lower(),
                "s": str(row.get("Speaker",   "")).strip(),
            })
        del ds
        gc.collect()
        with open(out_path, "w") as f:
            json.dump(rows, f)
        logger.info(f"[MELD] Raw cache saved → {out_path} ({len(rows)} rows)")
    except Exception as e:
        logger.warning(f"[MELD] Pre-download failed: {e} — MELD will be skipped.")

# MELD 7 labels → GoEmotions 28 (direct matches)
_MELD_TO_GOEMOTION: dict = {
    "anger":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "joy",
    "neutral":  "neutral",
    "sadness":  "sadness",
    "surprise": "surprise",
}

_MELD_VALENCE: dict = {
    "joy":      0.80,
    "surprise": 0.15,
    "neutral":  0.00,
    "anger":   -0.75,
    "disgust": -0.70,
    "fear":    -0.65,
    "sadness": -0.70,
}


def _meld_build_cdm(
    history: list,          # list of dicts: {valence, intent_state, speaker, goe_dist}
    current_valence: float,
    current_intent: int,
    current_speaker: str,
    current_text: str,
) -> np.ndarray:
    """
    Build a real 40-dim CDM context vector from MELD conversation history.
    Uses the same slot layout as context_engine_service so the meta-learner
    trains on vectors that look like real inference-time CDM output.
    """
    ctx = np.zeros(CDM_CTX_DIM, dtype=np.float32)
    n   = len(history)

    # ── CDM intent one-hot [0:15] ──────────────────────────────────────────
    ctx[current_intent] = 1.0

    # ── state_residency [15] — how long in this state ─────────────────────
    streak = 1
    for h in reversed(history):
        if h["intent_state"] == current_intent:
            streak += 1
        else:
            break
    ctx[CTX_RESIDENCY] = min(streak / max(n + 1, 1), 1.0)

    # ── transition_path [16:19] — last 3 intent indices / N_CDM_STATES ────
    recent = [h["intent_state"] for h in history[-3:]]
    for i, s in enumerate(recent):
        ctx[CTX_TRANSITION.start + i] = s / N_CDM_STATES

    # ── entry_abruptness [19] — sudden valence jump ────────────────────────
    prev_val = history[-1]["valence"] if history else 0.0
    ctx[CTX_ABRUPTNESS] = min(abs(current_valence - prev_val), 1.0)

    # ── topic_coherence [20] — stable intent = high coherence ─────────────
    if n > 0:
        same = sum(1 for h in history[-5:] if h["intent_state"] == current_intent)
        ctx[CTX_COHERENCE] = same / min(n, 5)
    else:
        ctx[CTX_COHERENCE] = 0.5

    # ── emotion_entropy [21] — diversity of previous labels ───────────────
    if n > 0:
        from collections import Counter
        ec     = Counter(h["intent_state"] for h in history[-5:])
        total  = sum(ec.values())
        probs  = [c / total for c in ec.values()]
        ent    = -sum(p * np.log(p + 1e-9) for p in probs)
        ctx[CTX_ENTROPY] = float(np.clip(ent / np.log(N_CDM_STATES), 0.0, 1.0))

    # ── speaker_divergence [22] ────────────────────────────────────────────
    if history:
        ctx[CTX_SPK_DIVERGENCE] = float(history[-1]["speaker"] != current_speaker)

    # ── velocity [23] and acceleration [24] ───────────────────────────────
    velocity = current_valence - prev_val
    ctx[CTX_VELOCITY] = float(np.clip(velocity, -1.0, 1.0))
    if len(history) >= 2:
        prev_velocity = history[-1]["valence"] - history[-2]["valence"]
        ctx[CTX_ACCELERATION] = float(np.clip(velocity - prev_velocity, -1.0, 1.0))

    # ── valence history [25:28] ────────────────────────────────────────────
    all_vals = [h["valence"] for h in history] + [current_valence]
    ctx[CTX_HIST_POS] = float(np.mean([v > 0.2  for v in all_vals]))
    ctx[CTX_HIST_NEU] = float(np.mean([abs(v) <= 0.2 for v in all_vals]))
    ctx[CTX_HIST_NEG] = float(np.mean([v < -0.2 for v in all_vals]))

    # ── topic_resonance [28] — reuse coherence as proxy ───────────────────
    ctx[CTX_RESONANCE] = ctx[CTX_COHERENCE]

    # ── volatility [29] — std of valence trajectory ────────────────────────
    if len(all_vals) > 1:
        ctx[CTX_VOLATILITY] = float(np.clip(np.std(all_vals), 0.0, 1.0))

    # ── current_valence [30] ───────────────────────────────────────────────
    ctx[CTX_CURR_VALENCE] = float(np.clip(current_valence, -1.0, 1.0))

    # ── message_length [31] (raw chars — normalized by build_feature_vector)
    ctx[CTX_MSG_LENGTH] = float(len(current_text))

    # ── latency_ms [32] — 0 (not available in MELD) ───────────────────────
    ctx[CTX_LATENCY_MS] = 0.0

    # ── HMM slots [33:39] — moderate confidence since we have real context ─
    # No live HMM in training; use mid-range values so ctx_weight is meaningful
    ctx[CTX_HMM_CONF]    = 0.65
    ctx[CTX_HMM_ENTROPY] = 0.40
    ctx[CTX_HMM_EMISSION] = -1.0
    ctx[CTX_HMM_NEXT3.start]     = 0.50
    ctx[CTX_HMM_NEXT3.start + 1] = 0.30
    ctx[CTX_HMM_NEXT3.start + 2] = 0.20

    # ── intent_stability [39] — same as residency ─────────────────────────
    ctx[CTX_INTENT_STAB] = ctx[CTX_RESIDENCY]

    return ctx


def extract_meld_features(
    vader_analyzer,
    bert_analyzer,
    goe_analyzer,
    max_utterances: int = 1500,
) -> tuple:
    """
    Build (X, y, has_cdm) from MELD with REAL conversation context.

    Unlike EmpatheticDialogues (topic-level label) or GoEmotions (single sentence),
    MELD provides multi-turn conversations where each utterance's CDM vector is
    built from the actual preceding messages in the same dialogue.

    This teaches the meta-learner:
      - "Previous messages were angry → current ambiguous message = probably anger"
      - "Context is neutral → trust NLP signal directly"
      - When context genuinely changes the answer vs when NLP is sufficient

    has_cdm = True for all MELD samples with at least 1 prior utterance,
    so the CDM block is NOT zeroed during training — the context path gets
    real gradient signal for the first time.
    """
    cache_path = MODEL_PATH.parent / "meld_features_cache.pkl"
    cache_key  = f"meld_ctx_v1_{max_utterances}_{FEATURE_DIM}"
    empty      = np.empty((0, FEATURE_DIM), dtype=np.float32)

    logger.info(f"[MELD] Starting feature extraction (max_utterances={max_utterances})...")
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                X, y, has_cdm = cached["data"]
                logger.info(f"[MELD] Cache hit — {len(y)} utterances with real context.")
                return np.array(X, dtype=np.float32) if X else empty, y, has_cdm
        except Exception as e:
            logger.warning(f"[MELD] Cache load failed: {e} — recomputing.")

    # ── Pass 1: load raw MELD rows ────────────────────────────────────────────
    # Raw rows are pre-downloaded by _download_meld_raw() before NLP models load.
    # If the raw file exists, read from it. Otherwise attempt a live load (may OOM).
    raw_json_path = MODEL_PATH.parent / "meld_raw_cache.json"
    raw_rows = []   # (dialogue_id, utterance_id, text, emotion_label, speaker)
    if raw_json_path.exists():
        try:
            with open(raw_json_path) as f:
                data = json.load(f)
            for r in data:
                raw_rows.append((r["d"], r["u"], r["t"], r["e"], r["s"]))
            logger.info(f"[MELD] Loaded {len(raw_rows)} rows from pre-downloaded raw cache.")
        except Exception as e:
            logger.warning(f"[MELD] Raw cache read failed: {e} — trying live load.")
            raw_rows = []

    if not raw_rows:
        # Download MELD train CSV directly — far faster than load_dataset which
        # regenerates from audio/video source files at ~1.5 rows/s (34 min for 3K rows).
        import urllib.request
        import zipfile
        import csv
        import io
        CSV_URLS = [
            # GitHub raw — original MELD repo
            "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
            # HuggingFace data dir alternatives
            "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/data/train_sent_emo.csv",
            "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/train_sent_emo.csv",
        ]
        csv_bytes = None
        for url in CSV_URLS:
            try:
                logger.info(f"[MELD] Trying CSV download: {url}")
                with urllib.request.urlopen(url, timeout=120) as resp:
                    size_mb = int(resp.headers.get("Content-Length", 0)) / 1e6
                    logger.info(f"[MELD] Downloading ({size_mb:.1f} MB)...")
                    csv_bytes = resp.read()
                logger.info(f"[MELD] Download complete ({len(csv_bytes)/1e6:.1f} MB).")
                break
            except Exception as e:
                logger.warning(f"[MELD] {url} failed: {e}")

        if not csv_bytes:
            logger.warning("[MELD] All CSV URLs failed — skipping MELD.")
            return empty, [], []

        try:
            # Handle zip or plain CSV
            if csv_bytes[:2] == b'PK':
                with zipfile.ZipFile(io.BytesIO(csv_bytes)) as zf:
                    csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                    text_data = zf.read(csv_name).decode("utf-8")
            else:
                text_data = csv_bytes.decode("utf-8")

            reader = csv.DictReader(io.StringIO(text_data))
            for row in reader:
                raw_rows.append((
                    str(row.get("Dialogue_ID", "")),
                    int(row.get("Utterance_ID", 0)),
                    str(row.get("Utterance",    "")).strip(),
                    str(row.get("Emotion",      "neutral")).lower(),
                    str(row.get("Speaker",      "")).strip(),
                ))
            logger.info(f"[MELD] Parsed {len(raw_rows)} raw rows from CSV.")
            save_data = [{"d": d, "u": u, "t": t, "e": e, "s": s}
                         for d, u, t, e, s in raw_rows]
            with open(raw_json_path, "w") as f:
                json.dump(save_data, f)
            logger.info(f"[MELD] Raw cache saved → {raw_json_path}")
        except Exception as e:
            logger.warning(f"[MELD] CSV parse failed: {e} — skipping.")
            return empty, [], []

    # Group and sort by dialogue
    dialogues: dict = {}
    for did, uid, text, emo, spk in raw_rows:
        dialogues.setdefault(did, []).append((uid, text, emo, spk))
    for did in dialogues:
        dialogues[did].sort(key=lambda x: x[0])

    # Flatten to ordered list
    all_rows = []   # (dialogue_id, uid, text, emo, speaker)
    for did, utt_list in dialogues.items():
        for uid, text, emo, spk in utt_list:
            all_rows.append((did, uid, text, emo, spk))
    all_rows = all_rows[:max_utterances]
    all_texts = [r[2] for r in all_rows]

    logger.info(f"[MELD] {len(all_rows)} utterances across {len(dialogues)} dialogues. Running batched NLP...")
    t0 = time.time()

    total_texts = len(all_texts)
    vader_outs = []
    _chk = {int(total_texts * p) for p in (0.25, 0.50, 0.75)}
    for i, text in enumerate(all_texts):
        try:
            vader_outs.append({f"vader_{k}": v for k, v in _vader(vader_analyzer, text).items()})
        except Exception:
            vader_outs.append({})
        if i + 1 in _chk:
            logger.info(f"  [MELD/VADER] {i+1}/{total_texts} ({int((i+1)/total_texts*100)}%)")

    bert_outs = _run_batch(bert_analyzer, all_texts, label="MELD/BERT") if callable(bert_analyzer) else [{} for _ in all_texts]
    goe_outs  = _run_batch(goe_analyzer,  all_texts, label="MELD/GoE")  if callable(goe_analyzer)  else [{} for _ in all_texts]

    logger.info(f"[MELD] NLP done in {time.time()-t0:.0f}s. Building CDM vectors from conversation history...")

    # ── Pass 2: build features with real context ───────────────────────────
    # Accumulate conversation history as we process each dialogue in order
    conv_history: dict = {}   # dialogue_id → list of history dicts

    features, labels, has_cdm_list = [], [], []
    _cdm_total = len(all_rows)
    _cdm_chk = {int(_cdm_total * p) for p in (0.25, 0.50, 0.75)}

    for _cdm_i, ((did, uid, text, meld_label, speaker), vader_out, bert_out, goe_out) in enumerate(zip(
        all_rows, vader_outs, bert_outs, goe_outs
    )):
        if _cdm_i + 1 in _cdm_chk:
            logger.info(f"  [MELD/CDM] {_cdm_i+1}/{_cdm_total} ({int((_cdm_i+1)/_cdm_total*100)}%) — {len(features)} vectors built so far")
        if not text:
            continue

        goemo_label = _MELD_TO_GOEMOTION.get(meld_label, "neutral")
        valence     = _MELD_VALENCE.get(meld_label, 0.0)
        intent_st   = _LABEL_TO_INTENT.get(goemo_label, (0, 0, 0))[0]

        history = conv_history.get(did, [])

        # First utterance in a conversation: no real context yet
        if not history:
            ctx        = np.zeros(CDM_CTX_DIM, dtype=np.float32)
            ctx[intent_st] = 1.0
            ctx[CTX_CURR_VALENCE] = valence
            ctx[CTX_MSG_LENGTH]   = float(len(text))
            ctx[CTX_HMM_CONF]     = 0.30   # low confidence — no history yet
            prior      = [0.0] * PRIOR_DIM
            real_ctx   = False
        else:
            ctx      = _meld_build_cdm(history, valence, intent_st, speaker, text)
            prior    = history[-1]["goe_dist"]   # previous utterance's GoE distribution
            real_ctx = True

        fv = build_feature_vector(
            {"vader": vader_out, "basic_bert": bert_out, "go_emotions": goe_out},
            context_vector=ctx,
            trajectory_prior=prior,
        )
        features.append(fv.flatten())
        labels.append(goemo_label)
        has_cdm_list.append(real_ctx)

        # Update history for next utterance
        goe_dist = [float(goe_out.get(e, 0.0)) for e in EMOTION_LABELS]
        conv_history.setdefault(did, []).append({
            "valence":      valence,
            "intent_state": intent_st,
            "speaker":      speaker,
            "goe_dist":     goe_dist,
        })

    real_count = sum(has_cdm_list)
    logger.info(
        f"[MELD] Built {len(features)} feature vectors — "
        f"{real_count} with real context ({100*real_count//max(len(features),1)}%), "
        f"{len(features)-real_count} first-utterance (no context)."
    )

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"cache_key": cache_key, "data": (features, labels, has_cdm_list)}, f)
        logger.info(f"[MELD] Cache saved → {cache_path}")
    except Exception as e:
        logger.warning(f"[MELD] Could not save cache: {e}")

    return np.array(features, dtype=np.float32) if features else empty, labels, has_cdm_list


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

def _get_analyzers(device: int):
    logger.info("Loading analyzers transiently into RAM...")
    torch.set_num_threads(1)
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


# ── Live data ───────────────────────────────────────────────────────────────────

def fetch_live_data(vader, bert, goe) -> tuple:
    """
    Fetch verified samples from PostgreSQL.
    Context is set to zeros (cold) — no conversation history available for SQL rows.
    """
    X, y = [], []
    try:
        engine = create_engine(DATABASE_URL)
        with engine.begin() as conn:
            try:
                conn.execute(text(
                    "ALTER TABLE emotion_analysis "
                    "ADD COLUMN IF NOT EXISTS ground_truth_emotion VARCHAR(50);"
                ))
            except Exception as e:
                logger.debug(f"  [SQL] Schema repair: {e}")

            rows = conn.execute(text(f"""
                WITH ranked AS (
                    SELECT m.message_id, m.text, a.ground_truth_emotion,
                           m.conversation_id, m.timestamp
                    FROM emotion_analysis a
                    JOIN messages m ON a.message_id = m.message_id
                    WHERE a.is_verified = TRUE
                )
                SELECT text, ground_truth_emotion
                FROM ranked
                ORDER BY timestamp DESC
                LIMIT {MAX_EMPATHETIC_SAMPLES}
            """)).fetchall()

        if not rows:
            return [], []

        logger.info(f"  [SQL] Found {len(rows)} verified live samples.")
        for text_content, label in rows:
            vs  = {f"vader_{k}": v for k, v in _vader(vader, text_content).items()}
            bs  = _run(bert, text_content)
            gs  = _run(goe,  text_content)
            ctx = build_synthetic_context_vector(mode="cold")
            fv  = build_feature_vector(
                {"vader": vs, "basic_bert": bs, "go_emotions": gs},
                context_vector=ctx[:CDM_CTX_DIM],
                trajectory_prior=ctx[CDM_CTX_DIM:],
            )
            X.append(fv.flatten())
            y.append(label)

        return X, y
    except Exception as e:
        logger.error(f"  [SQL] Failed to fetch live data: {e}")
        return [], []


# ── Main training cycle ─────────────────────────────────────────────────────────

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
    logger.info(f"═══ Starting cycle at {datetime.datetime.utcnow():%H:%M:%S UTC} ═══")

    prev_meta: dict = {}
    if META_PATH.exists():
        try:
            with open(META_PATH) as f:
                prev_meta = json.load(f)
        except Exception:
            pass

    is_bootstrap = not MODEL_PATH.exists()


    device_int = 0 if torch.cuda.is_available() else -1
    vader, bert, goe = _get_analyzers(device_int)

    if is_bootstrap:
        # ── One-time bootstrap from EmpatheticDialogues ────────────────────────
        CACHE_PATH = MODEL_PATH.parent / "dataset_features_cache.pkl"
        DATASET_ID = "empathetic_dialogues_v1"

        _va = lambda t: {f"vader_{k}": v for k, v in _vader(vader, t).items()}
        _ba = lambda t: _run(bert, t)
        _ga = lambda t: _run(goe,  t)

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
                X_tr, y_tr, gs_tr = cached["train"]
                X_v,  y_v,  _     = cached["val"]
                X_te, y_te, _     = cached["test"]
                logger.info(
                    f"Bootstrap cache loaded — "
                    f"{len(X_tr)} train / {len(X_v)} val / {len(X_te)} test samples."
                )
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Rebuilding.")
                CACHE_PATH.unlink(missing_ok=True)
                del vader, bert, goe
                gc.collect()
                return run_one_cycle(reload_callback)
        else:
            logger.info(
                f"No model found — bootstrapping from EmpatheticDialogues "
                f"(MAX_EMPATHETIC_SAMPLES={MAX_EMPATHETIC_SAMPLES})..."
            )
            X_tr, y_tr, gs_tr = extract_empathetic_dialogues_features(_va, _ba, _ga, split="train")
            X_v,  y_v,  _     = extract_empathetic_dialogues_features(_va, _ba, _ga, split="val")
            X_te, y_te, _     = extract_empathetic_dialogues_features(_va, _ba, _ga, split="test")

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
                    "train": (list(X_tr), y_tr, gs_tr),
                    "val":   (list(X_v),  y_v,  None),
                    "test":  (list(X_te), y_te, None),
                }, f)

            # Convert to lists for the augmentation step below
            X_tr, X_v, X_te = list(X_tr), list(X_v), list(X_te)

        has_cdm_tr: list = [False] * len(X_tr)
        n_ed = len(X_tr)

        # ── Synthetic augmentation for the 5 missing GoEmotions classes ────────
        X_syn, y_syn, gs_syn = load_synthetic_features(_va, _ba, _ga)
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
        # Teaches the meta-learner to trust NLP models on clear-cut single sentences.
        # Split 70/15/15: train gets 3× oversample; val+test get GoEmotions representation
        # so the final per-class metrics cover all 28 classes (not just ED classes).
        from sklearn.model_selection import train_test_split as _tts
        X_goe_d, y_goe_d, gs_goe_d = extract_goemotions_direct_features(_va, _ba, _ga)
        n_goe_d = 0
        n_goe_val = 0
        n_goe_test = 0
        if len(X_goe_d) > 0:
            X_goe_arr = np.array(X_goe_d, dtype=np.float32)
            # 70% train, 30% holdout — then split holdout 50/50 into val/test
            X_g_tr, X_g_hold, y_g_tr, y_g_hold = _tts(
                X_goe_arr, y_goe_d, test_size=0.30, random_state=42, stratify=y_goe_d
            )
            X_g_val, X_g_te, y_g_val, y_g_te = _tts(
                X_g_hold, y_g_hold, test_size=0.50, random_state=42, stratify=y_g_hold
            )

            # Log per-class split breakdown
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

            # Train: 3× oversample to compete with ED volume
            n_goe_d    = len(X_g_tr) * 3
            X_tr       = list(X_tr) + list(X_g_tr) * 3
            y_tr       = y_tr + list(y_g_tr) * 3
            gs_tr      = gs_tr + gs_goe_d[:len(X_g_tr)] * 3   # approx — gs unused after feature-build
            has_cdm_tr = has_cdm_tr + [False] * n_goe_d

            # Val + test: add GoEmotions samples so all 28 classes appear in evaluation
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
        # MELD provides 13K utterances from real conversations where each
        # utterance's CDM vector is built from the actual preceding messages.
        # has_cdm=True for these samples — the context path gets real gradient
        # signal for the first time (not synthetic noise or all-zeros).
        X_meld, y_meld, has_cdm_meld = extract_meld_features(_va, _ba, _ga)
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
        # ── Continuous learning from database ──────────────────────────────────
        from sklearn.model_selection import train_test_split as _tts

        X_live, y_live = fetch_live_data(vader, bert, goe)
        X_rel,  y_rel  = load_relabeled_conversations()

        X_all = X_live * 3 + list(X_rel) * 3
        y_all = y_live * 3 + list(y_rel) * 3

        if len(X_all) < MIN_DB_SAMPLES:
            logger.info(
                f"Only {len(X_all)} DB samples (need ≥ MIN_DB_SAMPLES={MIN_DB_SAMPLES}). "
                f"Skipping cycle — waiting for more verified data."
            )
            del vader, bert, goe
            gc.collect()
            return

        logger.info(f"  [DB] {len(X_all)} samples available for continuous learning.")

        X_tr, X_v, y_tr, y_v = _tts(X_all, y_all, test_size=0.20, random_state=42)
        X_te, y_te = X_v, y_v  # val doubles as test for gate consistency
        gs_tr      = [{label: 1.0} for label in y_tr]
        has_cdm_tr = [False] * len(X_tr)
        dataset_composition = {
            "Live DB samples (3×)":      len(X_live) * 3,
            "Relabeled conversations (3×)": len(X_rel) * 3,
        }
        logger.info(f"DB cycle: {len(X_tr)} train / {len(X_v)} val samples.")

    del vader, bert, goe
    gc.collect()
    logger.info("Transient analysers purged.")

    # ── Dataset composition summary (shown BEFORE training starts) ────────────
    comp_rows = {src: cnt for src, cnt in dataset_composition.items() if cnt > 0}
    comp_rows["─── Totals ───"] = ""
    comp_rows["  Train samples"] = len(X_tr)
    comp_rows["  Val samples  "] = len(X_v)
    comp_rows["  Test samples "] = len(X_te)
    logger.log_stats("Dataset Composition (before training)", comp_rows)

    # ── Filters ────────────────────────────────────────────────────────────────
    dist_before = Counter(y_tr).most_common(5)
    logger.log_stats("Pre-Filter Distribution (Top 5)", dict(dist_before))

    n_before = len(X_tr)
    if is_bootstrap:
        # EmpatheticDialogues is a curated dataset — trust its labels; skip outlier filter
        # (GoEmotions model disagrees with ~50% of ED labels due to different annotation
        # guidelines, which would remove half the bootstrap data for no good reason)
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

    if not X_tr:
        logger.error("No samples after filtering. Aborting.")
        return

    # ── Build numpy arrays ─────────────────────────────────────────────────────
    X_tr_arr = np.vstack([np.array(fv).flatten() for fv in X_tr])
    X_v_arr  = np.vstack([np.array(fv).flatten() for fv in X_v])
    X_te_arr = np.vstack([np.array(fv).flatten() for fv in X_te])

    # ── Train GatingEnsembleNet ────────────────────────────────────────────────
    logger.info("⚡ PHASE: GatingEnsembleNet Training (v2)...")
    train_device = "cuda" if torch.cuda.is_available() else "cpu"

    wrapper = train_gating_network(
        X_tr=X_tr_arr,
        y_tr=y_tr,
        X_v=X_v_arr,
        y_v=y_v,
        classes=EMOTION_LABELS,
        has_cdm=np.array(has_cdm_tr, dtype=bool),
        n_epochs=80,
        batch_size=256,
        lr=5e-4,
        weight_decay=1e-4,
        load_balance_coeff=5e-4,
        patience=10,
        device=train_device,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
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
    for lbl in EMOTION_LABELS:          # fixed order, same as EMOTION_LABELS list
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

    # ── Accuracy gate + deploy ─────────────────────────────────────────────────
    deployed = test_acc >= ACCURACY_GATE

    if deployed:
        tmp = MODEL_PATH.with_suffix(".tmp.pkl")
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

        # Publish reload signal in a separate try so a stats-write failure never
        # suppresses the signal — central_responder must reload regardless.
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


# ── Standalone entry point (TRAINER_EXTERNAL=true mode) ────────────────────────

if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    logger.info("Trainer running in standalone container mode (reload via Redis pub/sub).")
    logger.log_stats("Trainer Configuration", {
        "RETRAIN_INTERVAL_SECONDS": RETRAIN_INTERVAL,
        "ACCURACY_GATE":            ACCURACY_GATE,
        "MAX_EMPATHETIC_SAMPLES":   MAX_EMPATHETIC_SAMPLES,
        "MIN_DB_SAMPLES":           MIN_DB_SAMPLES,
        "MODEL_PATH":               str(MODEL_PATH),
        "RELOAD_CHANNEL":           RELOAD_CHANNEL,
    })
    while True:
        try:
            run_one_cycle(reload_callback=None)
        except Exception as e:
            logger.error(f"Standalone trainer cycle failed: {e}")
            traceback.print_exc()
        logger.info(f"Sleeping {RETRAIN_INTERVAL}s until next cycle...")
        time.sleep(RETRAIN_INTERVAL)
