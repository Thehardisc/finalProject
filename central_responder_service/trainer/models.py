"""
trainer/models.py — PyTorch GatingEnsembleNet training loop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from shared.constants import EMOTION_LABELS, ML_DIM, CDM_CTX_DIM
from shared.utils.logger import get_logger
from meta_learner import GatingEnsembleNet, GatingNetworkWrapper, D_MODEL, DROPOUT
from trainer.data.synthetic import pretrain_context_encoder

logger = get_logger("trainer")


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
    load_balance_coeff: float = 5e-4,
    patience: int = 10,
    device: str = None,
) -> GatingNetworkWrapper:
    """
    Train GatingEnsembleNet and return a sklearn-compatible GatingNetworkWrapper.

    Training strategy:
      - StandardScaler fitted on X_tr only (no leakage into val/test)
      - CrossEntropy with label_smoothing=0.10 + inverse-frequency class weights
      - Load-balance regularizer: λ·‖ᾱ − 1/3‖² prevents expert collapse
        (without it, GoEmotions tends to capture ~90% of gate weight)
      - OneCycleLR: fast convergence on small datasets
      - Early stopping on val macro-F1 (better metric than accuracy for 28 imbalanced classes)
      - Gradient clipping at 1.0 for training stability
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  [Trainer] Training on device={device}")

    # ── CDM masking: zero out CDM block for samples without real context ──────
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
    goe_to_class_idx = torch.tensor(
        [class_to_idx.get(lbl, -1) for lbl in EMOTION_LABELS],
        dtype=torch.long, device=device,
    )  # [28]  — -1 for labels absent from training classes

    nlp_anchor_coeff  = 0.30
    nlp_anchor_thresh = 0.75

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
            alpha_mean   = alpha.mean(dim=0)                         # [4]
            balance_loss = ((alpha_mean - uniform_gate) ** 2).sum()

            # NLP anchor: when GoEmotions is highly confident, steer logits toward
            # its top prediction.
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
