"""
train.py — Step 3: Train the ConversationLSTM on collected conversation sequences.

Usage:
    python3 train.py                          # default settings
    python3 train.py --epochs 200 --hidden 256 --layers 2
    python3 train.py --loss kl --lr 5e-4

Output (saved to models/checkpoints/):
    best_model.pt          — best checkpoint (lowest val loss)
    training_history.json  — loss curve per epoch
    model_config.json      — hyperparameters used
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.lstm    import ConversationLSTM
from models.dataset import load_sequence_dataset, make_loaders
from models.trainer import train, eval_epoch, top_k_accuracy
from features.schema import MSG_DIM, N_EMOTIONS

FEATURES_DIR  = Path(__file__).parent / "features"
CKPT_DIR      = Path(__file__).parent / "models" / "checkpoints"


def main():
    parser = argparse.ArgumentParser(description="Train ConversationLSTM")
    parser.add_argument("--epochs",  type=int,   default=100,  help="Max training epochs")
    parser.add_argument("--hidden",  type=int,   default=128,  help="LSTM hidden dim")
    parser.add_argument("--layers",  type=int,   default=2,    help="LSTM num layers")
    parser.add_argument("--dropout", type=float, default=0.3,  help="Dropout rate")
    parser.add_argument("--lr",      type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch",   type=int,   default=4,    help="Batch size")
    parser.add_argument("--patience",type=int,   default=20,   help="Early stopping patience")
    parser.add_argument("--loss",    choices=["mse", "kl"], default="mse")
    parser.add_argument("--device",  default="cpu", help="cpu / cuda / mps")
    args = parser.parse_args()

    print("=== STEP 3: TRAIN ConversationLSTM ===\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading sequence dataset...")
    try:
        train_ds, val_ds, test_ds = load_sequence_dataset(FEATURES_DIR)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    n_train = len(train_ds) if train_ds else 0
    n_val   = len(val_ds)   if val_ds   else 0
    n_test  = len(test_ds)  if test_ds  else 0
    print(f"  Train: {n_train}  Val: {n_val}  Test: {n_test} conversations")

    if n_train == 0:
        print("ERROR: No training data. Run collect.py and features/extract.py first.")
        sys.exit(1)

    # Reduce batch size if we have very few samples
    batch_size = min(args.batch, n_train)

    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )

    # ── Build model ────────────────────────────────────────────────────────────
    model = ConversationLSTM(
        input_dim=MSG_DIM,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        output_dim=N_EMOTIONS,
        dropout=args.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ConversationLSTM")
    print(f"  Input:   {MSG_DIM} dims/step")
    print(f"  Hidden:  {args.hidden}  Layers: {args.layers}  Dropout: {args.dropout}")
    print(f"  Output:  {N_EMOTIONS} dims/step (next emotion distribution)")
    print(f"  Params:  {n_params:,}")

    # ── Save config ────────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "input_dim":  MSG_DIM,
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "output_dim": N_EMOTIONS,
        "dropout":    args.dropout,
        "lr":         args.lr,
        "loss_fn":    args.loss,
        "epochs":     args.epochs,
        "patience":   args.patience,
        "n_train":    n_train,
        "n_val":      n_val,
        "n_test":     n_test,
    }
    with open(CKPT_DIR / "model_config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    # ── Train ──────────────────────────────────────────────────────────────────
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader or train_loader,   # fallback if no val split
        out_dir=CKPT_DIR,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        loss_fn=args.loss,
        device=args.device,
    )

    # ── Evaluate best model on test set ───────────────────────────────────────
    if test_loader and (CKPT_DIR / "best_model.pt").exists():
        print("\nLoading best checkpoint for final evaluation...")
        ckpt = torch.load(CKPT_DIR / "best_model.pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

        device = torch.device(args.device)
        model  = model.to(device)

        if args.loss == "kl":
            import torch.nn as nn
            criterion = lambda p, t: nn.KLDivLoss(reduction="batchmean")(torch.log(p + 1e-9), t)
        else:
            import torch.nn as nn
            criterion = nn.MSELoss()

        test_loss = eval_epoch(model, test_loader, criterion, device)
        top1_acc  = top_k_accuracy(model, test_loader, device, k=1)
        top3_acc  = top_k_accuracy(model, test_loader, device, k=3)

        print(f"\nTest results:")
        print(f"  Loss:      {test_loss:.4f}")
        print(f"  Top-1 acc: {top1_acc:.1%}  (predicted dominant = actual dominant)")
        print(f"  Top-3 acc: {top3_acc:.1%}  (predicted dominant in actual top-3)")

        # Append test results to config
        config["test_loss"]  = test_loss
        config["top1_acc"]   = top1_acc
        config["top3_acc"]   = top3_acc
        config["best_epoch"] = ckpt["epoch"]
        with open(CKPT_DIR / "model_config.json", "w") as fh:
            json.dump(config, fh, indent=2)

    # ── Deploy to central_responder_service/models/ ───────────────────────────
    best_ckpt = CKPT_DIR / "best_model.pt"
    if best_ckpt.exists():
        deploy_dir = Path(__file__).parent.parent / "central_responder_service" / "models"
        deploy_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy(best_ckpt, deploy_dir / "trajectory_lstm.pt")

        # Write trajectory_config.json with the trained architecture
        deploy_cfg = {
            "input_dim":  config["input_dim"],
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["num_layers"],
            "output_dim": config["output_dim"],
            "dropout":    config["dropout"],
        }
        with open(deploy_dir / "trajectory_config.json", "w") as fh:
            json.dump(deploy_cfg, fh, indent=2)

        print(f"\n✅ Deployed to {deploy_dir}/")
        print(f"   trajectory_lstm.pt      (input_dim={config['input_dim']})")
        print(f"   trajectory_config.json")
    else:
        print("\n⚠  No checkpoint found — model not deployed.")

    print(f"\n=== DONE — checkpoint saved to {CKPT_DIR}/ ===")


if __name__ == "__main__":
    main()
