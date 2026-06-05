"""
migrate_and_train.py — one-shot LSTM retraining after CDM_CTX_DIM 38→40 change.

What this does:
  1. Pads existing sequences.pkl from 77-dim to 79-dim (zeros at positions [77:79])
  2. Updates models/checkpoints/model_config.json
  3. Retrains ConversationLSTM with input_dim=79
  4. Deploys trajectory_lstm.pt + trajectory_config.json to
     central_responder_service/models/

Run inside the trainer_service container (or any env with PyTorch):
    cd /app
    python conversation_state_learner/migrate_and_train.py

Or from project root:
    docker compose run --rm trainer_service python conversation_state_learner/migrate_and_train.py
"""
import sys, os, json, pickle, shutil
import numpy as np
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

FEATURES_DIR = _HERE / "features"
CKPT_DIR     = _HERE / "models" / "checkpoints"
# The bind mount in docker-compose maps host ./central_responder_service/models → /app/models
# Write directly to /app/models so the host sees the new files immediately.
_env_model_path = os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")
DEPLOY_DIR   = Path(_env_model_path).parent

OLD_DIM = 77   # 28+7+4+38
NEW_DIM = 79   # 28+7+4+40

# ── Step 1: migrate sequences.pkl ─────────────────────────────────────────────

print("=== Step 1: Migrate sequences 77-dim → 79-dim ===")

seqs_path = FEATURES_DIR / "sequences.pkl"
if not seqs_path.exists():
    print(f"ERROR: {seqs_path} not found — run features/extract.py first.")
    sys.exit(1)

data = pickle.load(open(seqs_path, "rb"))
sequences    = data["sequences"]
seq_targets  = data["seq_targets"]

needs_migration = any(s.shape[1] == OLD_DIM for s in sequences if hasattr(s, "shape"))

if needs_migration:
    migrated = []
    for s in sequences:
        if s.shape[1] == OLD_DIM:
            pad = np.zeros((s.shape[0], NEW_DIM - OLD_DIM), dtype=np.float32)
            migrated.append(np.concatenate([s, pad], axis=1))
        elif s.shape[1] == NEW_DIM:
            migrated.append(s)
        else:
            print(f"  WARNING: unexpected dim {s.shape[1]}, skipping")
            migrated.append(s)
    data["sequences"] = migrated
    # keep a backup
    backup = seqs_path.with_suffix(".pkl.bak77")
    shutil.copy(seqs_path, backup)
    pickle.dump(data, open(seqs_path, "wb"))
    print(f"  Migrated {len(migrated)} conversations → {NEW_DIM}-dim")
    print(f"  Backup saved to {backup.name}")
else:
    print(f"  Sequences already {NEW_DIM}-dim, no migration needed.")

print()

# ── Step 2: train ─────────────────────────────────────────────────────────────

print("=== Step 2: Retrain ConversationLSTM (input_dim=79) ===\n")

import torch
from models.lstm    import ConversationLSTM
from models.dataset import load_sequence_dataset, make_loaders, ConversationDataset
from models.trainer import train, eval_epoch, top_k_accuracy
from features.schema import MSG_DIM, N_EMOTIONS
from features.meld   import load_meld_sequences

assert MSG_DIM == NEW_DIM, f"schema.py MSG_DIM={MSG_DIM}, expected {NEW_DIM}"

train_ds, val_ds, test_ds = load_sequence_dataset(FEATURES_DIR)
n_train = len(train_ds)
n_val   = len(val_ds)   if val_ds   else 0
n_test  = len(test_ds)  if test_ds  else 0
print(f"  Live pipeline sequences — Train: {n_train}  Val: {n_val}  Test: {n_test}")

# ── Augment with MELD (1,433 dialogues from Friends TV) ───────────────────────
print("\n  Loading NLP models for MELD extraction...")
import os as _os
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE",  "0")   # MELD needs network on first run
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VSA
from transformers import pipeline as _hfpipe
import torch as _torch

_device = 0 if _torch.cuda.is_available() else -1
_vader = _VSA()
_bert  = _hfpipe("text-classification",
                  model="j-hartmann/emotion-english-distilroberta-base",
                  top_k=None, device=_device)
_goe   = _hfpipe("text-classification",
                  model="bhadresh-savani/bert-base-go-emotion",
                  top_k=None, device=_device)

def _vader_fn(text):
    s = _vader.polarity_scores(text)
    return {"vader_neg": s["neg"], "vader_neu": s["neu"],
            "vader_pos": s["pos"], "vader_compound": s["compound"]}

def _bert_fn(text):
    try:
        return {r["label"]: r["score"] for r in _bert(text[:512])[0]}
    except Exception:
        return {}

def _goe_fn(text):
    try:
        return {r["label"]: r["score"] for r in _goe(text[:512])[0]}
    except Exception:
        return {}

meld_seqs, meld_tgts = load_meld_sequences(
    vader_fn=_vader_fn,
    bert_fn=_bert_fn,
    goe_fn=_goe_fn,
    max_dialogues=1200,   # train+val splits → ~1,000 dialogues
)

# Free models before training
del _bert, _goe, _vader
import gc; gc.collect()
print(f"  MELD sequences: {len(meld_seqs)} dialogues loaded")

# Merge: first 80 % of MELD into train, remainder into val
_meld_split = int(len(meld_seqs) * 0.8)
meld_train_seqs, meld_train_tgts = meld_seqs[:_meld_split],  meld_tgts[:_meld_split]
meld_val_seqs,   meld_val_tgts   = meld_seqs[_meld_split:],  meld_tgts[_meld_split:]

# Wrap in ConversationDataset and merge with live-pipeline data
from torch.utils.data import ConcatDataset

if meld_train_seqs:
    meld_train_ds = ConversationDataset(
        meld_train_seqs, meld_train_tgts,
        ["meld"] * len(meld_train_seqs),
    )
    train_ds = ConcatDataset([train_ds, meld_train_ds]) if train_ds is not None else meld_train_ds

if meld_val_seqs:
    meld_val_ds = ConversationDataset(
        meld_val_seqs, meld_val_tgts,
        ["meld"] * len(meld_val_seqs),
    )
    val_ds = ConcatDataset([val_ds, meld_val_ds]) if val_ds is not None else meld_val_ds

n_train = len(train_ds) if train_ds is not None else 0
n_val   = len(val_ds)   if val_ds   is not None else 0
n_test  = len(test_ds)  if test_ds  is not None else 0
print(f"  Combined — Train: {n_train}  Val: {n_val}  Test: {n_test} conversations")

if n_train == 0:
    print("ERROR: No training data found.")
    sys.exit(1)

batch_size = min(4, n_train)
train_loader, val_loader, test_loader = make_loaders(
    train_ds, val_ds, test_ds, batch_size=batch_size
)

HIDDEN  = 128
LAYERS  = 2
DROPOUT = 0.2

model = ConversationLSTM(
    input_dim=MSG_DIM,
    hidden_dim=HIDDEN,
    num_layers=LAYERS,
    output_dim=N_EMOTIONS,
    dropout=DROPOUT,
)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model: input={MSG_DIM} hidden={HIDDEN} layers={LAYERS} params={n_params:,}")

CKPT_DIR.mkdir(parents=True, exist_ok=True)

config = {
    "input_dim":  MSG_DIM,
    "hidden_dim": HIDDEN,
    "num_layers": LAYERS,
    "output_dim": N_EMOTIONS,
    "dropout":    DROPOUT,
    "lr":         1e-3,
    "loss_fn":    "mse",
    "epochs":     100,
    "patience":   20,
    "n_train":    n_train,
    "n_val":      n_val,
    "n_test":     n_test,
}

history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader or train_loader,
    out_dir=CKPT_DIR,
    epochs=100,
    lr=1e-3,
    patience=20,
    loss_fn="mse",
    device="cpu",
)

# ── Step 3: evaluate ──────────────────────────────────────────────────────────

best_ckpt = CKPT_DIR / "best_model.pt"
if test_loader and best_ckpt.exists():
    print("\n=== Step 3: Evaluate on test set ===")
    import torch.nn as nn
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    criterion = nn.MSELoss()
    test_loss = eval_epoch(model, test_loader, criterion, torch.device("cpu"))
    top1 = top_k_accuracy(model, test_loader, torch.device("cpu"), k=1)
    top3 = top_k_accuracy(model, test_loader, torch.device("cpu"), k=3)
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Top-1 acc: {top1:.1%}   Top-3 acc: {top3:.1%}")
    config.update({
        "test_loss": test_loss,
        "top1_acc": top1,
        "top3_acc": top3,
        "best_epoch": ckpt["epoch"],
    })

with open(CKPT_DIR / "model_config.json", "w") as fh:
    json.dump(config, fh, indent=2)

# ── Step 4: deploy ─────────────────────────────────────────────────────────────

print("\n=== Step 4: Deploy to central_responder_service/models/ ===")

if best_ckpt.exists():
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_ckpt, DEPLOY_DIR / "trajectory_lstm.pt")
    deploy_cfg = {
        "input_dim":  config["input_dim"],
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "output_dim": config["output_dim"],
        "dropout":    config["dropout"],
    }
    with open(DEPLOY_DIR / "trajectory_config.json", "w") as fh:
        json.dump(deploy_cfg, fh, indent=2)
    print(f"  ✅ trajectory_lstm.pt  → {DEPLOY_DIR}/")
    print(f"  ✅ trajectory_config.json (input_dim={config['input_dim']})")
else:
    print("  ⚠  No checkpoint — deploy skipped.")

print("\n=== DONE ===")
print("Next: rebuild central_responder_service to pick up the new model.")
print("  docker compose up --build central_responder_service trainer_service -d")
