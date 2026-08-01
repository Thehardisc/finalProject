from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from shared.constants import EMOTION_LABELS, ML_DIM, CDM_CTX_DIM
from shared.utils.logger import get_logger
from shared.utils.progress import TrainingProgress
from meta_learner import GatingEnsembleNet, GatingNetworkWrapper, D_MODEL, DROPOUT

logger = get_logger("trainer")


def pretrain_context_encoder(
    model,
    X_ctx: np.ndarray,
    y_labels: list,
    classes: list,
    has_cdm: np.ndarray = None,
    n_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
):
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
    prog = TrainingProgress(logger, task="ctx_pretrain", total=n_epochs, unit="epoch", log_every=5)
    for epoch in range(n_epochs):
        model.train()
        logits = model.ctx_prior_head(model.enc_ctx_expert(X_ctx_t))
        loss   = F.cross_entropy(logits, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        prog.step(metrics={"loss": loss.item()})

    prog.done(note="enc_ctx_expert primed")
    model.eval()
    return model


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
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"  [Trainer] Training on device={device}")

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

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_v_s   = scaler.transform(X_v)

    if has_cdm is not None:
        if no_cdm.any():
            X_tr_s[no_cdm, CDM_SLICE] = 0.0
        X_v_s[:, CDM_SLICE] = 0.0

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_tr_idx = np.array([class_to_idx[y]          for y in y_tr])
    y_v_idx  = np.array([class_to_idx.get(y, 0)   for y in y_v])

    counts   = np.bincount(y_tr_idx, minlength=len(classes)).clip(1)
    weights  = (1.0 / counts) ** 0.5
    weights  = (weights / weights.mean()).clip(0.25, 4.0)
    cw_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    pin_mem   = device == "cuda"
    n_workers = 2 if len(X_tr_s) > 5000 else 0
    ds     = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr_idx, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False,
                        pin_memory=pin_mem, num_workers=n_workers,
                        persistent_workers=(n_workers > 0))

    model = GatingEnsembleNet(n_classes=len(classes), d=D_MODEL, dropout=DROPOUT).to(device)

    logger.info("  [CtxPretrain] Pre-training context encoder on context block...")
    model = pretrain_context_encoder(
        model, X_tr_s, y_tr, classes,
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
    n_train = len(y_tr_idx)
    label_smoothing = 0.05 if n_train > 5000 else 0.10
    criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=label_smoothing)
    logger.info(f"  [Trainer] label_smoothing={label_smoothing} (n_train={n_train})")

    goe_to_class_idx = torch.tensor(
        [class_to_idx.get(lbl, -1) for lbl in EMOTION_LABELS],
        dtype=torch.long, device=device,
    )

    nlp_anchor_coeff  = 0.3
    nlp_anchor_thresh = 0.50

    uniform_gate  = torch.ones(5, device=device) / 5.0
    X_v_t         = torch.tensor(X_v_s, dtype=torch.float32, device=device)
    best_f1       = -1.0
    best_state    = None
    no_improve    = 0

    use_amp    = device == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None
    amp_ctx    = torch.cuda.amp.autocast if use_amp else nullcontext
    logger.info(f"  [Trainer] AMP (mixed precision): {'enabled' if use_amp else 'disabled'}")

    n_train_samples = len(y_tr_idx)
    prog = TrainingProgress(logger, task="meta_gating", total=n_epochs, unit="epoch")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            with amp_ctx():
                logits, alpha = model(xb)
                ce_loss    = criterion(logits, yb)

                alpha_mean   = alpha.mean(dim=0)
                balance_loss = ((alpha_mean - uniform_gate) ** 2).sum()

                goe_raw            = xb[:, 11:39]
                goe_conf, goe_ridx = goe_raw.max(dim=1)
                goe_cidx           = goe_to_class_idx[goe_ridx]
                anchor_mask        = (goe_conf > nlp_anchor_thresh) & (goe_cidx >= 0)
                anchor_loss = (
                    F.cross_entropy(logits[anchor_mask], goe_cidx[anchor_mask])
                    if anchor_mask.any() else torch.tensor(0.0, device=device)
                )

                loss = ce_loss + load_balance_coeff * balance_loss + nlp_anchor_coeff * anchor_loss

            if use_amp:
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            logits_v, alpha_v = model(X_v_t)
            preds_v = logits_v.argmax(dim=-1).cpu().numpy()

        val_f1      = f1_score(y_v_idx, preds_v, average='macro', zero_division=0)
        alpha_mean_ = alpha_v.mean(dim=0).cpu().numpy()
        avg_loss    = epoch_loss / len(loader)
        vad_str = f" vad:{alpha_mean_[3]:.3f}" if len(alpha_mean_) > 4 else ""
        ctx_str = f"{vad_str} ctx:{alpha_mean_[-1]:.3f}]" if len(alpha_mean_) > 3 else "]"

        prog.step(
            metrics={"loss": avg_loss, "val_F1": val_f1},
            note=f"α=[vader:{alpha_mean_[0]:.3f} bert:{alpha_mean_[1]:.3f} goe:{alpha_mean_[2]:.3f}{ctx_str}",
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

    overall_sps = (n_train_samples * prog.count / prog.elapsed) if prog.elapsed > 0 else 0
    prog.done(metrics={"samples_per_s": int(overall_sps)})

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info(f"  [Trainer] Best checkpoint restored (val F1={best_f1:.4f})")

    model.eval()
    return GatingNetworkWrapper(model=model, classes=classes, scaler=scaler)
