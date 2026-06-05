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
import gc
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
from sqlalchemy import create_engine, text
from shared.utils.logger import get_logger
import redis as redis_sync

logger = get_logger("trainer")

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
from sklearn.metrics import accuracy_score, f1_score

from shared.constants import EMOTION_LABELS, VADER_KEYS, BERT_LABELS, FEATURE_DIM, CONTEXT_DIM, CDM_CTX_DIM, PRIOR_DIM, ML_DIM, N_CDM_STATES
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
            alpha_mean   = alpha.mean(dim=0)                         # [3]
            balance_loss = ((alpha_mean - uniform_gate) ** 2).sum()
            loss         = ce_loss + load_balance_coeff * balance_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        # ── Validation every 5 epochs ─────────────────────────────────────────
        if (epoch + 1) % 5 == 0:
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
    'prepared':    'optimism',   'embarrassed':  'embarrassment',
    'excited':     'excitement', 'annoyed':      'annoyance',
    'lonely':      'sadness',    'surprised':    'surprise',
    'furious':     'anger',      'disappointed': 'disappointment',
    'caring':      'caring',     'trusting':     'approval',
    'disgusted':   'disgust',    'anticipating': 'optimism',
    'anxious':     'nervousness','nostalgic':    'realization',
    'confident':   'pride',      'content':      'relief',
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

    total = len(rows)
    logger.info(f"  [EmpDialogues] Building {split} features from {total} situations...")
    t0 = time.time()

    features, labels, gs_list = [], [], []
    for i, row in enumerate(rows):
        if i > 0 and i % 500 == 0:
            elapsed  = time.time() - t0
            rate     = i / elapsed
            eta_s    = int((total - i) / rate) if rate > 0 else 0
            logger.info(
                f"  [EmpDialogues] {split} {i}/{total} "
                f"({i * 100 // total}%)  "
                f"{rate:.1f} samples/s  ETA {eta_s}s"
            )

        text        = str(row.get('situation', '')).strip()
        raw_emotion = str(row.get('emotion', 'neutral')).lower()
        goemo_label = _EMPATHETIC_TO_GOEMOTION.get(raw_emotion, 'neutral')

        if not text:
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


def extract_meld_features(
    vader_analyzer,
    bert_analyzer,
    goe_analyzer,
    max_utterances: int = 5000,
) -> tuple:
    """
    Build (X [N, FEATURE_DIM], y [N]) from MELD for meta-learner training.

    MELD has 13,708 utterances from Friends TV across 7 emotion classes that
    map directly to GoEmotions labels.  CDM context is synthetic (same noise
    as GoEmotions training samples) since there is no live pipeline data.
    """
    try:
        from datasets import load_dataset
        meld = load_dataset("declare-lab/MELD")
    except Exception as e:
        logger.warning(f"MELD load failed: {e} — skipping.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    features, labels = [], []
    n_processed = 0

    for split in ("train", "validation"):
        if split not in meld:
            continue
        for row in meld[split]:
            if n_processed >= max_utterances:
                break
            text = str(row.get("Utterance", "")).strip()
            if not text:
                continue

            goemo_label = _MELD_TO_GOEMOTION.get(
                row.get("Emotion", "neutral").lower(), "neutral"
            )

            try:
                vs  = {f"vader_{k}": v
                       for k, v in _vader(vader_analyzer, text).items()}
                bs  = _run(bert_analyzer, text)
                gs  = _run(goe_analyzer,  text)
            except Exception:
                continue

            # Synthetic context — same augmentation as GoEmotions training
            ctx = build_synthetic_context_vector(label=goemo_label, mode="train")
            fv  = build_feature_vector(
                {"vader": vs, "basic_bert": bs, "go_emotions": gs},
                context_vector=ctx[:CDM_CTX_DIM],
                trajectory_prior=ctx[CDM_CTX_DIM:],
            )
            features.append(fv.flatten())
            labels.append(goemo_label)
            n_processed += 1

        if n_processed >= max_utterances:
            break

    logger.info(f"  [MELD] Extracted {len(features)} utterances.")
    return (
        np.array(features, dtype=np.float32) if features
        else np.empty((0, FEATURE_DIM), dtype=np.float32),
        labels,
    )


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
) -> None:
    prev_acc = prev_meta.get("test_accuracy")
    delta    = (new_acc - prev_acc) if prev_acc is not None else None

    stats = {
        "Previous Accuracy":  f"{prev_acc:.4f}" if prev_acc is not None else "N/A",
        "New Accuracy (test)": f"{new_acc:.4f}  {_bar(new_acc)}",
        "New F1 (macro)":      f"{new_f1:.4f}  {_bar(new_f1)}",
        "Samples Trained":     n_train,
        "Samples Filtered":    n_filtered,
        "Deployment":          "✅ DEPLOYED" if deployed else "❌ REJECTED (accuracy/regression)",
    }
    if delta is not None:
        direction    = "↑" if delta >= 0 else "↓"
        stats["Delta"] = f"{direction} {delta*100:+.2f}%"

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

        if CACHE_PATH.exists():
            logger.info("Loading cached bootstrap features...")
            try:
                with open(CACHE_PATH, "rb") as f:
                    cached = pickle.load(f)
                if (cached.get("dataset_id") != DATASET_ID or
                        cached.get("feature_dim") != FEATURE_DIM):
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
            _va = lambda t: {f"vader_{k}": v for k, v in _vader(vader, t).items()}
            _ba = lambda t: _run(bert, t)
            _ga = lambda t: _run(goe,  t)

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
                    "train": (list(X_tr), y_tr, gs_tr),
                    "val":   (list(X_v),  y_v,  None),
                    "test":  (list(X_te), y_te, None),
                }, f)

            # Convert to lists for the augmentation step below
            X_tr, X_v, X_te = list(X_tr), list(X_v), list(X_te)

        has_cdm_tr: list = [False] * len(X_tr)

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
        logger.info(f"DB cycle: {len(X_tr)} train / {len(X_v)} val samples.")

    del vader, bert, goe
    gc.collect()
    logger.info("Transient analysers purged.")

    # ── Filters ────────────────────────────────────────────────────────────────
    dist_before = Counter(y_tr).most_common(5)
    logger.log_stats("Pre-Filter Distribution (Top 5)", dict(dist_before))

    n_before                      = len(X_tr)
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

    print_report(prev_meta, test_acc, test_f1, len(X_tr), n_filtered, deployed)


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
