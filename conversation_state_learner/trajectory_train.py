#!/usr/bin/env python3
"""trajectory_train.py — Train EmotionalDialogueEncoder (EDE)."""

import argparse
import json

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import torch.nn.functional as F

ROOT  = Path(__file__).parent.parent
CACHE = ROOT / ".cache"
sys.path.insert(0, str(ROOT / "central_responder_service"))
from trajectory.model import EmotionalDialogueEncoder, PHASES, N_PHASES, PHASE_IDX

MELD_PATH   = CACHE / "meld_raw_cache.json"
EMPAT_CACHE = CACHE / "empathetic_raw_cache.json"
MODEL_OUT   = CACHE / "ede_model.pt"
CONFIG_OUT  = CACHE / "ede_config.json"

EMOTION_LABELS_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
NEUTRAL_IDX = EMOTION_LABELS_28.index("neutral")

_EMOTION_VALENCE = {
    "anger": -0.80, "annoyance": -0.60, "disapproval": -0.70, "disgust": -0.80,
    "disappointment": -0.60, "embarrassment": -0.50, "fear": -0.70, "grief": -0.90,
    "nervousness": -0.50, "remorse": -0.70, "sadness": -0.70,
    "admiration": 0.80, "amusement": 0.60, "approval": 0.60, "caring": 0.70,
    "curiosity": 0.30, "desire": 0.50, "excitement": 0.80, "gratitude": 0.80,
    "joy": 0.90, "love": 0.90, "optimism": 0.70, "pride": 0.70,
    "relief": 0.60, "neutral": 0.00, "confusion": -0.10,
    "realization": 0.10, "surprise": 0.20,
}


# ── Phase derivation ──────────────────────────────────────────────────────────

def derive_phase(valence_window: List[float]) -> int:
    if len(valence_window) < 3:
        return PHASE_IDX["opening"]
    v           = np.array(valence_window[-8:], dtype=np.float32)
    net_delta   = float(v[-1] - v[0])
    volatility  = float(np.std(v))
    mean_recent = float(np.mean(v[-3:]))
    velocity    = float(v[-1] - v[-2]) if len(v) >= 2 else 0.0
    if volatility > 0.25 and abs(mean_recent) > 0.20:
        return PHASE_IDX["peak"]
    if abs(velocity) > 0.20:
        return PHASE_IDX["turning_point"]
    if net_delta < -0.15 and mean_recent < 0.0:
        return PHASE_IDX["escalation"]
    if net_delta > 0.15 and mean_recent >= -0.05:
        return PHASE_IDX["resolution"]
    if volatility < 0.05:
        return PHASE_IDX["sustained"]
    return PHASE_IDX["opening"]


def vec_valence(vec: np.ndarray) -> float:
    return float(sum(vec[i] * _EMOTION_VALENCE.get(e, 0.0)
                     for i, e in enumerate(EMOTION_LABELS_28)))


# ── Real GoE inference ────────────────────────────────────────────────────────

def run_goe_model_on_texts(
    texts: List[str],
    cache_path: Optional[Path] = None,
    batch_size: int = 64,
) -> List[np.ndarray]:
    """Run bhadresh-savani/bert-base-go-emotion on a list of utterances."""
    if cache_path is not None and cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            print(f"  GoE inference cache hit: {cache_path.name} ({len(texts)} vectors)")
            return list(arr)
        print(f"  GoE cache size mismatch ({arr.shape[0]} vs {len(texts)}) — re-running")

    from transformers import pipeline as hf_pipeline

    print(f"  Loading GoE model (bhadresh-savani/bert-base-go-emotion)...")
    goe = hf_pipeline(
        "text-classification",
        model="bhadresh-savani/bert-base-go-emotion",
        top_k=None,
        device=-1,
        batch_size=batch_size,
    )

    results = []
    total = len(texts)
    for start in range(0, total, batch_size * 4):
        chunk = texts[start: start + batch_size * 4]
        preds = goe(chunk, truncation=True, max_length=512)
        for p in preds:
            d = {item["label"]: item["score"] for item in p}
            results.append(np.array([d.get(e, 0.0) for e in EMOTION_LABELS_28], dtype=np.float32))
        print(f"  GoE inference: {min(start + batch_size * 4, total)}/{total}", end="\r")

    print()
    if cache_path is not None:
        np.save(cache_path, np.stack(results))
        print(f"  Cached GoE results → {cache_path.name}")
    return results


# ── Data loading ──────────────────────────────────────────────────────────────

def load_meld_sequences(path: Path) -> List[Tuple[np.ndarray, np.ndarray, List[float]]]:
    with open(path) as f:
        rows = json.load(f)

    by_dialog: Dict[str, List] = defaultdict(list)
    for row in rows:
        by_dialog[row["d"]].append(row)

    all_texts:     List[str] = []
    all_positions: List[Tuple] = []
    for dialog_id, utts in by_dialog.items():
        for i, utt in enumerate(sorted(utts, key=lambda r: r["u"])):
            text = utt.get("t", "").strip()
            if text:
                all_texts.append(text)
                all_positions.append((dialog_id, i))

    print(f"MELD: {len(by_dialog)} dialogs, {len(all_texts)} utterances — running GoE...")
    goe_vecs = run_goe_model_on_texts(all_texts, cache_path=CACHE / "meld_goe_cache.npy")

    goe_by_pos: Dict = {}
    for pos, vec in zip(all_positions, goe_vecs):
        goe_by_pos[pos] = vec

    sequences = []
    for dialog_id, utts in by_dialog.items():
        utts_sorted = sorted(utts, key=lambda r: r["u"])
        vecs = [goe_by_pos.get((dialog_id, i)) for i in range(len(utts_sorted))]
        vecs = [v for v in vecs if v is not None]
        if len(vecs) < 2:
            continue
        arr = np.stack(vecs)
        X = arr[:-1]
        Y = arr[1:]
        V = [vec_valence(v) for v in arr[1:]]
        sequences.append((X, Y, V))

    total_pairs = sum(len(x) for x, _, _ in sequences)
    print(f"MELD: {len(sequences)} sequences, {total_pairs} step pairs (real GoE)")
    return sequences


def load_empathetic_sequences(path: Path) -> List[Tuple[np.ndarray, np.ndarray, List[float]]]:
    with open(path) as f:
        rows = json.load(f)

    by_conv: Dict[str, List] = defaultdict(list)
    for row in rows:
        cid = row.get("conv_id") or row.get("d") or row.get("c", "")
        by_conv[cid].append(row)

    all_texts:     List[str] = []
    all_positions: List[Tuple] = []
    for conv_id, utts in by_conv.items():
        sorted_utts = sorted(utts, key=lambda r: r.get("utt_idx", r.get("u", 0)))
        for i, utt in enumerate(sorted_utts):
            text = (utt.get("utterance") or utt.get("t") or utt.get("text") or "").strip()
            if text:
                all_texts.append(text)
                all_positions.append((conv_id, i))

    print(f"EmpatheticDialogues: {len(by_conv)} convs, {len(all_texts)} utterances — running GoE...")
    goe_vecs = run_goe_model_on_texts(all_texts, cache_path=CACHE / "empathetic_goe_cache.npy")

    goe_by_pos: Dict = {}
    for pos, vec in zip(all_positions, goe_vecs):
        goe_by_pos[pos] = vec

    sequences = []
    for conv_id, utts in by_conv.items():
        sorted_utts = sorted(utts, key=lambda r: r.get("utt_idx", r.get("u", 0)))
        vecs = [goe_by_pos.get((conv_id, i)) for i in range(len(sorted_utts))]
        vecs = [v for v in vecs if v is not None]
        if len(vecs) < 2:
            continue
        arr = np.stack(vecs)
        X = arr[:-1]
        Y = arr[1:]
        V = [vec_valence(v) for v in arr[1:]]
        sequences.append((X, Y, V))

    total_pairs = sum(len(x) for x, _, _ in sequences)
    print(f"EmpatheticDialogues: {len(sequences)} sequences, {total_pairs} step pairs (real GoE)")
    return sequences


def build_windows(
    sequences: List[Tuple[np.ndarray, np.ndarray, List[float]]],
    max_window: int,
    neutral_keep_rate: float = 0.40,
) -> List[dict]:
    """Build fixed-window training samples."""
    samples = []
    for X, Y, V in sequences:
        T = len(X)
        for t in range(T):
            target  = Y[t]
            is_neutral_dominant = int(target.argmax()) == NEUTRAL_IDX and target[NEUTRAL_IDX] > 0.50

            if is_neutral_dominant and random.random() > neutral_keep_rate:
                continue

            start  = max(0, t - max_window)
            hist   = X[start: t]
            vwin   = V[max(0, t - 7): t + 1]
            phase  = derive_phase(vwin)
            length = len(hist)

            if length < max_window:
                pad  = np.zeros((max_window - length, 28), dtype=np.float32)
                hist = np.concatenate([pad, hist], axis=0)
                mask = np.array([True] * (max_window - length) + [False] * length, dtype=bool)
            else:
                hist = hist[-max_window:]
                mask = np.zeros(max_window, dtype=bool)

            samples.append({
                "history":      hist,
                "target":       target,
                "phase":        phase,
                "padding_mask": mask,
            })
    return samples


# ── Training ──────────────────────────────────────────────────────────────────

def topk_acc(pred: torch.Tensor, tgt: torch.Tensor, k: int) -> float:
    tgt_cls = tgt.argmax(dim=-1)
    topk    = pred.topk(k, dim=-1).indices
    return (topk == tgt_cls.unsqueeze(-1)).any(dim=-1).float().mean().item()


def train(
    samples: List[dict],
    epochs:     int   = 60,
    lr:         float = 3e-4,
    batch_size: int   = 128,
    max_window: int   = 12,
    d_model:    int   = 64,
    val_frac:   float = 0.10,
    device:     str   = "cpu",
) -> EmotionalDialogueEncoder:

    random.shuffle(samples)
    n_val       = max(1, int(len(samples) * val_frac))
    val_samples = samples[:n_val]
    trn_samples = samples[n_val:]

    phase_counts = np.bincount([s["phase"] for s in trn_samples], minlength=N_PHASES).astype(np.float32)
    phase_counts = np.where(phase_counts == 0, 1.0, phase_counts)
    phase_w      = phase_counts.sum() / (N_PHASES * phase_counts)
    phase_w      = np.clip(phase_w, 0.20, 5.0)
    phase_w_t    = torch.tensor(phase_w, dtype=torch.float32, device=device)

    model = EmotionalDialogueEncoder(max_window=max_window, d_model=d_model).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(epochs // 3, 15), T_mult=1, eta_min=lr * 0.01,
    )

    best_top1  = 0.0
    best_state = None

    emo_weights = torch.ones(28, device=device) * 1.5
    emo_weights[NEUTRAL_IDX] = 0.5

    def run_batch(batch):
        H = torch.tensor(np.stack([s["history"] for s in batch]),      dtype=torch.float32, device=device)
        T = torch.tensor(np.stack([s["target"]  for s in batch]),      dtype=torch.float32, device=device)
        P = torch.tensor([s["phase"]             for s in batch],       dtype=torch.long,    device=device)
        M = torch.tensor(np.stack([s["padding_mask"] for s in batch]), dtype=torch.bool,    device=device)
        prior, phase_logits = model(H, padding_mask=M)
        log_prior = torch.log(prior.clamp(min=1e-8))
        weighted_kl = -(T * log_prior * emo_weights.unsqueeze(0)).sum(dim=-1).mean()
        ce_loss = F.cross_entropy(phase_logits, P, weight=phase_w_t)
        loss    = 0.7 * weighted_kl + 0.3 * ce_loss
        return loss, prior, T, phase_logits, P

    print(f"\nTrain: {len(trn_samples)} | Val: {len(val_samples)} | device: {device} | d_model: {d_model}")

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(trn_samples)
        trn_loss, n_b = 0.0, 0

        for i in range(0, len(trn_samples), batch_size):
            batch = trn_samples[i: i + batch_size]
            if not batch:
                continue
            opt.zero_grad()
            loss, *_ = run_batch(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trn_loss += loss.item()
            n_b      += 1

        sched.step()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            all_prior, all_tgt, all_ph_pred, all_ph_true = [], [], [], []
            val_loss = 0.0

            with torch.no_grad():
                for i in range(0, len(val_samples), batch_size):
                    batch = val_samples[i: i + batch_size]
                    if not batch:
                        continue
                    loss, prior, tgt, ph_logits, ph_true = run_batch(batch)
                    val_loss += loss.item()
                    all_prior.append(prior)
                    all_tgt.append(tgt)
                    all_ph_pred.append(ph_logits.argmax(dim=-1))
                    all_ph_true.append(ph_true)

            all_prior = torch.cat(all_prior)
            all_tgt   = torch.cat(all_tgt)
            top1      = topk_acc(all_prior, all_tgt, 1)
            top3      = topk_acc(all_prior, all_tgt, 3)
            ph_pred   = torch.cat(all_ph_pred).cpu().numpy()
            ph_true   = torch.cat(all_ph_true).cpu().numpy()
            phase_acc = float((ph_pred == ph_true).mean())
            neutral_pct = float((all_prior.argmax(dim=-1) == 27).float().mean())

            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"trn={trn_loss/max(n_b,1):.3f}  "
                f"val={val_loss/max(len(val_samples)//batch_size,1):.3f}  "
                f"top1={top1:.3f}  top3={top3:.3f}  "
                f"phase={phase_acc:.3f}  "
                f"neutral%={neutral_pct:.2%}"
                + ("  ← COLLAPSE" if neutral_pct > 0.60 else "")
            )

            if top1 > best_top1:
                best_top1  = top1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"\nDone — best top1={best_top1:.4f}")
    return model, best_top1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=60)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--hidden",  type=int,   default=64)
    parser.add_argument("--window",  type=int,   default=12)
    parser.add_argument("--no-empathetic", action="store_true")
    args = parser.parse_args()

    CACHE.mkdir(exist_ok=True)
    all_sequences = []

    if MELD_PATH.exists():
        all_sequences.extend(load_meld_sequences(MELD_PATH))
    else:
        print(f"MELD cache not found at {MELD_PATH}")

    if not args.no_empathetic and EMPAT_CACHE.exists():
        all_sequences.extend(load_empathetic_sequences(EMPAT_CACHE))
    elif not EMPAT_CACHE.exists():
        print(f"Empathetic cache not found at {EMPAT_CACHE}")

    if not all_sequences:
        print("ERROR: no training data found.")
        sys.exit(1)

    total_pairs = sum(len(x) for x, _, _ in all_sequences)
    print(f"\nTotal: {len(all_sequences)} sequences, {total_pairs} step pairs")

    samples = build_windows(all_sequences, max_window=args.window)
    print(f"Windows: {len(samples)} training samples")

    phase_dist = np.bincount([s["phase"] for s in samples], minlength=N_PHASES)
    for i, p in enumerate(PHASES):
        print(f"  {p:15s}: {phase_dist[i]:6d}  ({100*phase_dist[i]/max(len(samples),1):.1f}%)")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("Training on CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Training on Apple MPS")

    model, top1 = train(
        samples,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.hidden,
        max_window=args.window,
        device=device,
    )

    torch.save({"model_state": model.state_dict()}, MODEL_OUT)

    config = {
        "n_emotions":  28,
        "n_phases":    N_PHASES,
        "d_model":     args.hidden,
        "n_heads":     4,
        "n_layers":    2,
        "max_window":  args.window,
        "dropout":     0.15,
        "top1_acc":    round(top1, 4),
        "arch":        "EmotionalDialogueEncoder",
    }
    with open(CONFIG_OUT, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved: {MODEL_OUT}")
    print(f"       {CONFIG_OUT}")


if __name__ == "__main__":
    main()
