"""Training loop for ConversationLSTM."""

import json
import sys

import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root → shared/

from shared.utils.logger import get_logger
from shared.utils.progress import TrainingProgress

logger = get_logger("csl_trainer")





def _masked_loss(criterion, preds, targets, lengths):
    """Compute loss only on non-padded timesteps."""
    B, T, D = preds.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=preds.device)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    preds_flat   = preds[mask]
    targets_flat = targets[mask]

    return criterion(preds_flat, targets_flat)


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x, y, lengths, _ in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)

        optimizer.zero_grad()
        preds, _ = model(x, lengths=lengths)
        loss = _masked_loss(criterion, preds, y, lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for x, y, lengths, _ in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        preds, _ = model(x, lengths=lengths)
        loss = _masked_loss(criterion, preds, y, lengths)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def top_k_accuracy(model, loader, device, k: int = 3) -> float:
    """At each timestep, is the highest-predicted emotion in the top-k actual emotions?"""
    model.eval()
    correct = 0
    total   = 0

    for x, y, lengths, _ in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        preds, _ = model(x, lengths=lengths)

        B, T, D = preds.shape
        for b in range(B):
            L = int(lengths[b].item())
            for t in range(L):
                pred_top  = preds[b, t].topk(1).indices.item()
                true_topk = y[b, t].topk(k).indices.tolist()
                if pred_top in true_topk:
                    correct += 1
                total += 1

    return correct / max(total, 1)


def train(
    model,
    train_loader,
    val_loader,
    out_dir:    Path,
    epochs:     int   = 100,
    lr:         float = 1e-3,
    patience:   int   = 15,
    loss_fn:    str   = "mse",
    device:     str   = "cpu",
) -> dict:
    """Full training run with early stopping and checkpointing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    model  = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 3, min_lr=1e-5
    )

    if loss_fn == "kl":
        criterion = nn.KLDivLoss(reduction="batchmean")
        # KL expects log-probs as input; wrap model output
        def _criterion(preds, targets):
            return criterion(torch.log(preds + 1e-9), targets)
    else:
        _criterion = nn.MSELoss()

    best_val   = float("inf")
    patience_c = 0
    history    = {"train": [], "val": [], "epoch": []}

    prog = TrainingProgress(logger, task="conversation_lstm", total=epochs,
                            unit="epoch", log_every=10)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, _criterion, device)
        val_loss   = eval_epoch(model, val_loader,   _criterion, device) if val_loader else train_loss

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val   = val_loss
            patience_c = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "hidden_dim":  model.hidden_dim,
                "num_layers":  model.num_layers,
            }, out_dir / "best_model.pt")
        else:
            patience_c += 1

        prog.step(
            metrics={"train": train_loss, "val": val_loss, "lr": current_lr},
            note="◀ best" if improved else "",
            force=improved or epoch == 1,
        )

        if patience_c >= patience:
            logger.info(f"[conversation_lstm] early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)")
            break

    prog.done(metrics={"best_val": best_val})

    with open(out_dir / "training_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    return history
