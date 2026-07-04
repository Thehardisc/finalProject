import os
import sys
import json
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from shared.constants import EMOTION_LABELS
from shared.utils.logger import get_logger

logger = get_logger("finetune_goe")

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
MODEL_DIR  = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")).parent
OUTPUT_DIR = MODEL_DIR / "samlowe_finetuned"
MAX_LEN    = 128
EPOCHS     = 3
BATCH_SIZE = 32
LR         = 2e-5


class GoEDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc    = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids":      self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "label":          self.labels[i],
        }


def _class_weights(labels: list) -> torch.Tensor:
    from collections import Counter
    counts = Counter(labels)
    total  = len(labels)
    n_cls  = len(EMOTION_LABELS)
    freq   = np.array([counts.get(i, 0) / max(total, 1) for i in range(n_cls)], dtype=np.float32)
    freq   = np.where(freq > 0, freq, freq[freq > 0].min())
    w      = 1.0 / freq
    w      = np.clip(w / w.mean(), 0.1, 10.0)
    return torch.tensor(w, dtype=torch.float32)


def main():
    if OUTPUT_DIR.exists():
        logger.info(f"Fine-tuned model already exists at {OUTPUT_DIR} — skipping.")
        return

    logger.info(f"Fine-tuning {MODEL_NAME} → {OUTPUT_DIR}")
    t0 = time.time()

    from datasets import load_dataset
    logger.info("Loading GoEmotions train split...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    id2label = ds.features["labels"].feature.int2str
    label2idx = {id2label(i): i for i in range(len(EMOTION_LABELS))}

    texts, labels = [], []
    for row in ds:
        if len(row["labels"]) == 1:
            lbl_str = id2label(row["labels"][0])
            if lbl_str in label2idx:
                texts.append(row["text"])
                labels.append(label2idx[lbl_str])

    logger.info(f"  {len(texts)} single-label training examples across {len(EMOTION_LABELS)} classes.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset   = GoEDataset(texts, labels, tokenizer)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"  Training device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(EMOTION_LABELS),
        ignore_mismatched_sizes=True,
    ).to(device)

    weights = _class_weights(labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optim   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS * len(loader))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for i, batch in enumerate(loader):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["label"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, lbl)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            total_loss += loss.item() * len(lbl)
            correct    += (out.logits.argmax(dim=-1) == lbl).sum().item()
            n          += len(lbl)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"  Epoch {epoch}/{EPOCHS}  step {i+1}/{len(loader)}  "
                    f"loss={total_loss/n:.4f}  acc={correct/n:.4f}  {elapsed:.0f}s"
                )
        logger.info(f"  Epoch {epoch} done — loss={total_loss/n:.4f}  acc={correct/n:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    meta = {
        "base_model":     MODEL_NAME,
        "epochs":         EPOCHS,
        "lr":             LR,
        "n_train":        len(texts),
        "wall_sec":       round(time.time() - t0, 1),
        "output_dir":     str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "finetune_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Fine-tuning complete in {time.time()-t0:.0f}s — model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
