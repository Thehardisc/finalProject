#!/usr/bin/env python3
"""train_sarcasm_classifier.py — Fine-tune DistilBERT for binary sarcasm/passive-aggression detection."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizerFast, get_linear_schedule_with_warmup


ROOT       = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))  # repo root → shared/

from shared.utils.logger import get_logger
from shared.utils.progress import TrainingProgress

logger = get_logger("sarcasm_train")

DATA_FILE    = ROOT / "training_data" / "sarcasm_labels.jsonl"
# Sincere positive-surface negatives (train-only) — counters the tweet_eval bias
# where clean enthusiastic positives read as irony. See qa_suite/test_sarcasm.py.
AUGMENT_FILE = ROOT / "training_data" / "sarcasm_sincere_augment.jsonl"
MODELS_DIR = ROOT.parent / "central_responder_service" / "models"
MODEL_OUT  = MODELS_DIR / "sarcasm_clf.pt"
CONFIG_OUT = MODELS_DIR / "sarcasm_clf_config.json"

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
DROPOUT    = 0.3
SEED       = 42

# ── Model ─────────────────────────────────────────────────────────────────────

class SarcasmClassifier(nn.Module):
    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(BASE_MODEL)
        hidden    = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(self.drop(cls)).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_text(record: dict) -> str:
    """Concatenate context messages + target with [SEP] as boundary marker."""
    parts = [c["text"].strip() for c in record.get("context", []) if c.get("text")]
    parts.append(record["text"].strip())
    return " [SEP] ".join(parts)


class SarcasmDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer: DistilBertTokenizerFast):
        self.records   = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec  = self.records[idx]
        text = build_text(rec)
        enc  = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(float(rec["is_sarcastic"]), dtype=torch.float32),
        }


# ── Data loading & splitting ──────────────────────────────────────────────────

def load_records(path: Path) -> List[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def conversation_split(
    records: List[dict],
    val_frac: float = 0.20,
    seed: int = SEED,
) -> Tuple[List[dict], List[dict]]:
    """Split by conversation_id so no conversation spans train and val."""
    by_conv: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_conv[r["conversation_id"]].append(r)

    conv_ids = list(by_conv.keys())
    rng = random.Random(seed)
    rng.shuffle(conv_ids)

    n_val    = max(1, int(len(conv_ids) * val_frac))
    val_ids  = set(conv_ids[:n_val])

    train = [r for r in records if r["conversation_id"] not in val_ids]
    val   = [r for r in records if r["conversation_id"] in val_ids]
    return train, val


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    acc       = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    thresholds = np.linspace(0.0, 1.0, 51)
    tprs, fprs = [], []
    for t in thresholds:
        p  = (probs >= t).astype(int)
        tprs.append(((p == 1) & (labels == 1)).sum() / max(1, (labels == 1).sum()))
        fprs.append(((p == 1) & (labels == 0)).sum() / max(1, (labels == 0).sum()))
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # np.trapz removed in NumPy 2.0
    auc = float(_trapz(tprs[::-1], fprs[::-1]))

    return dict(acc=acc, precision=precision, recall=recall, f1=f1, auc=auc,
                tp=tp, fp=fp, fn=fn, tn=tn)


def tune_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """Sweep thresholds [0.1, 0.9]; return (threshold, f1) that maximises F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.05):
        m = compute_metrics(labels, probs, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t  = float(t)
    return best_t, best_f1


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model: SarcasmClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: SarcasmClassifier,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Returns (loss, all_labels, all_probs)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        labels = batch["label"].to(device)
        total_loss += criterion(logits, labels).item()
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return (
        total_loss / len(loader),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


# ── Dry-run: dataset stats ────────────────────────────────────────────────────

def print_stats(records: List[dict]) -> None:
    n          = len(records)
    n_sarc     = sum(r["is_sarcastic"] for r in records)
    subtypes   = Counter(r["subtype"] for r in records)
    conv_ids   = set(r["conversation_id"] for r in records)

    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    lengths   = [
        len(tokenizer(build_text(r), truncation=False)["input_ids"])
        for r in records
    ]
    over_limit = sum(1 for l in lengths if l > MAX_LENGTH)

    print(f"\n{'='*60}")
    print(f"Dataset: {DATA_FILE}")
    print(f"{'='*60}")
    print(f"  Total records  : {n}")
    print(f"  Conversations  : {len(conv_ids)}")
    print(f"  Sarcastic      : {n_sarc}  ({100*n_sarc/n:.1f}%)")
    print(f"  Not sarcastic  : {n - n_sarc}  ({100*(n-n_sarc)/n:.1f}%)")
    print(f"  Subtype counts : {dict(subtypes)}")
    print(f"  Token lengths  : min={min(lengths)} mean={np.mean(lengths):.0f} "
          f"p95={np.percentile(lengths, 95):.0f} max={max(lengths)}")
    print(f"  Truncated msgs : {over_limit} / {n}  (>{MAX_LENGTH} tokens)")

    by_sub: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_sub[r["subtype"]].append(r)
    print(f"\n  Examples:")
    for sub, recs in by_sub.items():
        if sub == "none":
            continue
        for r in recs[:2]:
            ctx_str = " | ".join(c["text"][:40] for c in r.get("context", []))
            print(f"    [{sub}] ctx='{ctx_str}'  target='{r['text'][:60]}'")
            print(f"           reason: {r.get('reasoning', '')}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print dataset stats and exit — no training.")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--val-frac",   type=float, default=0.20,
                        help="Fraction of conversations held out for validation.")
    parser.add_argument("--patience",   type=int,   default=3,
                        help="Early-stopping patience (epochs without val F1 improvement).")
    parser.add_argument("--warmup-frac", type=float, default=0.10,
                        help="Fraction of total steps used for linear LR warmup.")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run relabel.py --mode sarcasm first.", file=sys.stderr)
        sys.exit(1)

    records = load_records(DATA_FILE)
    print(f"Loaded {len(records)} records from {DATA_FILE}")

    if args.dry_run:
        print_stats(records)
        return

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()         else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    train_recs, val_recs = conversation_split(records, val_frac=args.val_frac)

    if AUGMENT_FILE.exists():
        aug = load_records(AUGMENT_FILE)
        train_recs = train_recs + aug
        print(f"Augment: +{len(aug)} sincere-domain negatives (train only; val stays pure)")

    n_pos   = sum(r["is_sarcastic"] for r in train_recs)
    n_neg   = len(train_recs) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(device)

    print(f"Train: {len(train_recs)} msgs  |  Val: {len(val_recs)} msgs")
    print(f"Train positive rate: {n_pos}/{len(train_recs)} = {100*n_pos/len(train_recs):.1f}%")
    print(f"pos_weight: {pos_weight.item():.2f}")

    train_ds = SarcasmDataset(train_recs, tokenizer)
    val_ds   = SarcasmDataset(val_recs,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model     = SarcasmClassifier(dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_steps   = len(train_dl) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_frac)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1    = 0.0
    best_threshold = 0.5
    best_state     = None
    no_improve     = 0

    logger.info(f"Training for up to {args.epochs} epochs (patience={args.patience})")
    prog = TrainingProgress(logger, task="sarcasm_clf", total=args.epochs, unit="epoch")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, criterion, device)
        val_loss, val_labels, val_probs = evaluate(model, val_dl, criterion, device)

        threshold, val_f1 = tune_threshold(val_labels, val_probs)
        metrics = compute_metrics(val_labels, val_probs, threshold=threshold)

        prog.step(metrics={
            "train_loss": train_loss, "val_loss": val_loss,
            "F1": metrics["f1"], "AUC": metrics["auc"],
            "P": metrics["precision"], "R": metrics["recall"], "thr": threshold,
        })

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1    = val_f1
            best_threshold = threshold
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve     = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[sarcasm_clf] early stopping at epoch {epoch} "
                            f"(no improvement for {args.patience} epochs)")
                break

    if best_state is None:
        logger.warning("No best state found — saving current model.")
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, val_labels, val_probs = evaluate(model, val_dl, criterion, device)
    final_metrics = compute_metrics(val_labels, val_probs, threshold=best_threshold)

    prog.done(metrics={
        "F1": final_metrics["f1"], "AUC": final_metrics["auc"],
        "acc": final_metrics["acc"], "thr": best_threshold,
    })
    logger.info(f"Confusion: TP={final_metrics['tp']} FP={final_metrics['fp']} "
                f"FN={final_metrics['fn']} TN={final_metrics['tn']}")

    MODELS_DIR.mkdir(exist_ok=True)

    torch.save({"model_state": best_state}, MODEL_OUT)

    config = {
        "base_model":      BASE_MODEL,
        "max_length":      MAX_LENGTH,
        "hidden_size":     768,
        "dropout":         DROPOUT,
        "threshold":       best_threshold,
        "trained_at":      datetime.now().isoformat(),
        "epochs_run":      epoch,
        "val_f1":          round(final_metrics["f1"], 4),
        "val_auc":         round(final_metrics["auc"], 4),
        "val_precision":   round(final_metrics["precision"], 4),
        "val_recall":      round(final_metrics["recall"], 4),
        "n_train":         len(train_recs),
        "n_val":           len(val_recs),
        "positive_rate":   round(n_pos / len(train_recs), 4) if train_recs else 0.0,
        "pos_weight_used": round(pos_weight.item(), 4),
        "lr":              args.lr,
        "batch_size":      args.batch_size,
    }
    CONFIG_OUT.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print(f"\nSaved model  → {MODEL_OUT}")
    print(f"Saved config → {CONFIG_OUT}")


if __name__ == "__main__":
    main()
