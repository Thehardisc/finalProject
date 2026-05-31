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
  MAX_SAMPLES               (default: 2500)
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
RETRAIN_INTERVAL = int(os.environ.get("RETRAIN_INTERVAL_SECONDS", 1800))
ACCURACY_GATE    = float(os.environ.get("ACCURACY_GATE",          0.40))
MAX_SAMPLES      = int(os.environ.get("MAX_SAMPLES",              2500))
MODEL_PATH       = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
META_PATH        = MODEL_PATH.with_name("meta_weights_meta.json")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from shared.constants import EMOTION_LABELS, VADER_KEYS, BERT_LABELS, FEATURE_DIM, CONTEXT_DIM, ML_DIM, N_CDM_STATES
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
    n_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> GatingEnsembleNet:
    """
    Pre-train enc_ctx_expert + ctx_prior_head on context → emotion before
    full pipeline training. Forces the context encoder to learn emotion-predictive
    representations rather than noise, giving the Bayesian prior a useful starting point.

    Only trains enc_ctx_expert and ctx_prior_head — all other parameters frozen.
    """
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

    # Pre-train context encoder before full pipeline training
    logger.info("  [CtxPretrain] Pre-training context encoder on context block...")
    model = pretrain_context_encoder(model, X_tr, y_tr, classes, n_epochs=20, device=device)

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
    max_conversations: int = 2000,
) -> tuple:
    """
    Build (X [N, 77], y [N]) from EmpatheticDialogues with real HMM context history.

    For each conversation, runs HMM forward filtering through utterances,
    producing a real context vector rather than synthetic noise.
    """
    try:
        from datasets import load_dataset
        emp = load_dataset("empathetic_dialogues")
    except Exception as e:
        logger.warning(f"EmpatheticDialogues load failed: {e} — skipping.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    transmat, emissionprob = _load_hmm_params()
    n_states = N_CDM_STATES
    if transmat is None:
        logger.warning("HMM params unavailable — skipping EmpatheticDialogues.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    # Group utterances by conversation id
    convs: dict = {}
    for row in emp['train']:
        cid = row.get('conv_id', '')
        if cid not in convs:
            convs[cid] = []
        convs[cid].append(row)

    features, labels = [], []
    conv_list = list(convs.items())[:max_conversations]

    logger.info(f"  [EmpDialogues] Building context features from {len(conv_list)} conversations...")

    for conv_id, turns in conv_list:
        turns_sorted = sorted(turns, key=lambda t: int(t.get('utterance_idx', 0)))
        conv_emotion = turns_sorted[0].get('context', 'neutral').lower()
        goemo_label  = _EMPATHETIC_TO_GOEMOTION.get(conv_emotion, 'neutral')

        hmm_alpha   = np.ones(n_states, dtype=np.float64) / n_states
        val_hist    = []
        prev_val    = 0.0

        for i, turn in enumerate(turns_sorted):
            text = str(turn.get('utterance', '')).strip()
            if not text:
                continue

            # NLP features for this utterance
            try:
                vader_out = vader_analyzer(text) if callable(vader_analyzer) else {}
                bert_out  = bert_analyzer(text)  if callable(bert_analyzer)  else {}
                goe_out   = goe_analyzer(text)   if callable(goe_analyzer)   else {}
            except Exception:
                continue

            # Valence history for velocity
            cur_val  = float(vader_out.get('vader_compound', 0.0))
            val_hist.append(cur_val)
            velocity     = cur_val - prev_val
            acceleration = (velocity - (prev_val - val_hist[-3] if len(val_hist) >= 3 else 0.0))
            prev_val     = cur_val

            # CDM state from HMM forward variable
            cdm_state  = int(np.argmax(hmm_alpha))
            cdm_onehot = [0.0] * n_states
            cdm_onehot[cdm_state] = 1.0

            hmm_conf    = float(hmm_alpha.max())
            hmm_entropy = float(-np.sum(hmm_alpha * np.log(hmm_alpha + 1e-12)))
            top3_next   = sorted(transmat[cdm_state].tolist(), reverse=True)[:3]
            stab_count  = sum(1 for _ in range(min(i, 5)) if i > 0)  # approximate
            intent_stab = min(stab_count / 10.0, 1.0)

            # Build context vector (38-dim)
            ctx = (
                cdm_onehot +
                [min(i / 10.0, 1.0)] +   # residency
                [cdm_state / n_states, cdm_state / n_states, cdm_state / n_states] +  # transition path
                [0.0,         # abruptness (no prev state for first turn)
                 0.5,         # topic coherence (neutral approx)
                 0.5,         # emotion entropy (neutral approx)
                 0.0,         # speaker divergence (single speaker)
                 float(np.clip(velocity, -1, 1)),
                 float(np.clip(acceleration, -1, 1)),
                 _EMPATHETIC_VALENCE.get(conv_emotion, 0.0),  # hist valence from conv emotion
                 0.5,         # topic resonance
                 0.2,         # volatility
                 float(np.clip(cur_val, -1, 1)),
                 float(np.clip(len(text) / 500.0, 0, 1)),
                 0.1,         # latency
                 hmm_conf,
                 hmm_entropy,
                 0.7 if hmm_conf > 0.5 else 0.3,  # emission approx
                ] + top3_next +
                [intent_stab]
            )

            if len(ctx) != CONTEXT_DIM:
                continue  # skip malformed

            # ML block
            ml = []
            for k in VADER_KEYS:
                ml.append(float(vader_out.get(k, 0.0)))
            for k in BERT_LABELS:
                ml.append(float(bert_out.get(k, 0.0)))
            for k in EMOTION_LABELS:
                ml.append(float(goe_out.get(k, 0.0)))

            fv = np.array(ml + ctx, dtype=np.float32)
            if len(fv) != FEATURE_DIM:
                continue

            features.append(fv)
            labels.append(goemo_label)

            # Update HMM for next turn
            obs        = _empathetic_obs(conv_emotion)
            hmm_alpha  = _hmm_forward_step(hmm_alpha, transmat, emissionprob, obs)

    logger.info(f"  [EmpDialogues] Extracted {len(features)} conv-context samples.")
    return (np.array(features, dtype=np.float32) if features else np.empty((0, FEATURE_DIM), dtype=np.float32),
            labels)


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
    # Prevent HuggingFace from doing network version-check calls on every load.
    # Models are already cached after first download; offline mode avoids hangs.
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import pipeline as hf_pipeline

    vader = SentimentIntensityAnalyzer()
    bert  = hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=device,
    )
    goe   = hf_pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        return_all_scores=True,
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

def filter_outliers(X: list, y: list, goe_list: list):
    """Drop samples where GoEmotions gives < 5% confidence to the gold label."""
    cX, cy, removed = [], [], 0
    for fv, label, goe in zip(X, y, goe_list):
        if label not in EMOTION_LABELS or goe.get(label, 0.0) < 0.05:
            removed += 1
        else:
            cX.append(fv)
            cy.append(label)
    return cX, cy, removed


def filter_balance(X: list, y: list):
    """Cap any class at 3× the median class count."""
    if not y:
        return X, y
    counts = Counter(y)
    cap    = max(50, int(statistics.median(counts.values()) * 3))
    seen   = Counter()
    cX, cy = [], []
    for fv, label in zip(X, y):
        if seen[label] < cap:
            cX.append(fv)
            cy.append(label)
            seen[label] += 1
    removed = len(X) - len(cX)
    if removed:
        logger.info(f"  [Filter] Balance cap: removed {removed} samples (cap={cap}/class).")
    return cX, cy


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
                LIMIT {MAX_SAMPLES}
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
                context_vector=ctx,
            )
            X.append(fv.flatten())
            y.append(label)

        return X, y
    except Exception as e:
        logger.error(f"  [SQL] Failed to fetch live data: {e}")
        return [], []


# ── Main training cycle ─────────────────────────────────────────────────────────

def run_one_cycle(reload_callback) -> None:
    """
    Full build → filter → train → gate → deploy cycle.

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

    # ── Feature vector cache ───────────────────────────────────────────────────
    CACHE_PATH   = MODEL_PATH.parent / "dataset_features_cache.pkl"
    _fresh_build = not CACHE_PATH.exists()  # track before we create it
    if CACHE_PATH.exists():
        logger.info("Loading cached dataset features...")
        try:
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)

            if cached.get("feature_dim") != FEATURE_DIM:
                logger.warning(
                    f"Cache dim={cached.get('feature_dim')} ≠ current {FEATURE_DIM}. Invalidating."
                )
                CACHE_PATH.unlink(missing_ok=True)
                return run_one_cycle(reload_callback)

            X_tr, y_tr, gs_tr = cached["train"]
            X_v,  y_v,  _     = cached["val"]
            X_te, y_te, _     = cached["test"]

            device_int = 0 if torch.cuda.is_available() else -1
            vader, bert, goe  = _get_analyzers(device_int)
            logger.info("Features loaded from cache.")

        except Exception as e:
            logger.warning(f"Cache load failed: {e}. Rebuilding.")
            CACHE_PATH.unlink(missing_ok=True)
            return run_one_cycle(reload_callback)
    else:
        # ── Fresh build from GoEmotions ────────────────────────────────────────
        t0 = time.time()
        logger.info("Loading GoEmotions dataset...")
        try:
            from datasets import load_dataset
            ds = load_dataset("google-research-datasets/go_emotions", "simplified")
        except Exception as e:
            logger.error(f"Dataset load failed: {e}")
            return

        train_raw = list(ds["train"])[:MAX_SAMPLES]
        val_raw   = list(ds["validation"])[:MAX_SAMPLES // 5]
        test_raw  = list(ds["test"])[:MAX_SAMPLES // 5]
        logger.info(f"  [Extraction] {len(train_raw)} train / {len(val_raw)} val / {len(test_raw)} test samples")

        device_int = 0 if torch.cuda.is_available() else -1
        vader, bert, goe = _get_analyzers(device_int)

        def process(split: list, name: str, ctx_mode: str):
            X, y, gs_list = [], [], []
            for i, s in enumerate(split):
                if i % 100 == 0:
                    logger.info(
                        f"  [Trainer] {name} {i}/{len(split)} "
                        f"({i / len(split) * 100:.0f}%)"
                    )
                lids = s.get("labels", [])
                if not lids or lids[0] >= len(EMOTION_LABELS):
                    continue
                label    = EMOTION_LABELS[lids[0]]
                text_val = s["text"]
                vs  = {f"vader_{k}": v for k, v in _vader(vader, text_val).items()}
                bs  = _run(bert, text_val)
                gs  = _run(goe,  text_val)
                gs_list.append(gs)
                # Pass label + mode so context is correctly augmented per split
                ctx = build_synthetic_context_vector(label=label, mode=ctx_mode)
                X.append(
                    build_feature_vector(
                        {"vader": vs, "basic_bert": bs, "go_emotions": gs},
                        context_vector=ctx,
                    ).flatten()
                )
                y.append(label)

            pt = (time.time() - t0) / max(len(X), 1)
            logger.info(f"  [Trainer] {name}: {len(X)} samples, {pt*1000:.2f}ms/sample avg")
            return X, y, gs_list

        logger.info("Building feature vectors...")
        X_tr, y_tr, gs_tr = process(train_raw, "train", "train")
        X_v,  y_v,  _     = process(val_raw,   "val",   "val")
        X_te, y_te, _     = process(test_raw,  "test",  "cold")  # cold = no-context worst case

        logger.info("Saving feature cache...")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({
                "feature_dim": FEATURE_DIM,
                "train": (X_tr, y_tr, gs_tr),
                "val":   (X_v,  y_v,  None),
                "test":  (X_te, y_te, None),
            }, f)

    # ── Live supervised data augmentation ──────────────────────────────────────
    X_live, y_live = fetch_live_data(vader, bert, goe)
    if X_live:
        weight = 3
        X_tr.extend(X_live * weight)
        y_tr.extend(y_live * weight)
        gs_tr.extend([{}] * len(X_live) * weight)
        logger.info(
            f"  [Trainer] Augmented with {len(X_live)} live samples (weight={weight})"
        )

    # ── EmpatheticDialogues: real conversation context augmentation ────────────
    # Only on fresh build (analyzers still in RAM) to avoid re-running every cycle.
    if _fresh_build:
        logger.info("  [EmpDialogues] Extracting conversation-context features...")
        X_emp, y_emp = extract_empathetic_dialogues_features(
            vader_analyzer=lambda t: {f"vader_{k}": v for k, v in _vader(vader, t).items()},
            bert_analyzer=lambda t: _run(bert, t),
            goe_analyzer=lambda t: _run(goe, t),
            max_conversations=2000,
        )
        if len(X_emp) > 0:
            # Mix 40% EmpatheticDialogues with 60% GoEmotions in training set
            emp_weight = 2  # each EmpDialogues sample counts twice vs GoEmotions
            X_tr.extend([row for row in X_emp] * emp_weight)
            y_tr.extend(y_emp * emp_weight)
            # Set gold-label confidence=1.0 so filter_outliers keeps these samples
            gs_tr.extend([{label: 1.0} for label in y_emp] * emp_weight)
            logger.info(f"  [EmpDialogues] Added {len(X_emp)*emp_weight} conv-context samples to training set.")

    del vader, bert, goe
    gc.collect()
    logger.info("Transient analysers purged.")

    # ── Filters ────────────────────────────────────────────────────────────────
    dist_before = Counter(y_tr).most_common(5)
    logger.log_stats("Pre-Filter Distribution (Top 5)", dict(dist_before))

    n_before       = len(X_tr)
    X_tr, y_tr, n_out = filter_outliers(X_tr, y_tr, gs_tr)
    logger.info(f"  Layer 2 (outlier): removed {n_out}")
    X_tr, y_tr     = filter_balance(X_tr, y_tr)
    n_filtered     = n_before - len(X_tr)

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
            "MAX_SAMPLES":              MAX_SAMPLES,
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
