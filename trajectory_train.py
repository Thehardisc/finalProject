#!/usr/bin/env python3
"""
trajectory_train.py — Stage 3: Retrain ConversationLSTM on MELD 1,038 conversations.

Input:  .cache/meld_raw_cache.json   (9,989 utterances already downloaded)
Output: .cache/trajectory_lstm.pt    (mounted into central_responder_service)
        .cache/trajectory_config.json

Feature vector per utterance: 79 dims
  [  0: 27]  go_emotions    28 dims — soft distribution from MELD 7-class label
  [ 28: 34]  basic_bert      7 dims — soft distribution from MELD 7-class label
  [ 35: 38]  vader           4 dims — synthetic valence features
  [ 39: 78]  cdm_context    40 dims — built from real conversation history

Run from project root:
  python trajectory_train.py [--epochs N] [--lr F] [--hidden N]
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT  = Path(__file__).parent
CACHE = ROOT / ".cache"
sys.path.insert(0, str(ROOT / "central_responder_service"))
from trajectory.model import ConversationLSTM

MELD_PATH  = CACHE / "meld_raw_cache.json"
MODEL_OUT  = CACHE / "trajectory_lstm.pt"
CONFIG_OUT = CACHE / "trajectory_config.json"

# ── Label maps ────────────────────────────────────────────────────────────────
EMOTION_LABELS_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
GoE_IDX  = {e: i for i, e in enumerate(EMOTION_LABELS_28)}

BERT_LABELS_7 = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
BERT_IDX = {e: i for i, e in enumerate(BERT_LABELS_7)}

# MELD 7-class → GoEmotions 28 soft distribution (sums to 1.0)
MELD_TO_GoE: dict[str, dict[int, float]] = {
    "anger":    {GoE_IDX["anger"]: 0.50, GoE_IDX["annoyance"]: 0.25,
                 GoE_IDX["disapproval"]: 0.15, GoE_IDX["disgust"]: 0.10},
    "disgust":  {GoE_IDX["disgust"]: 0.55, GoE_IDX["disapproval"]: 0.25,
                 GoE_IDX["annoyance"]: 0.20},
    "fear":     {GoE_IDX["fear"]: 0.55, GoE_IDX["nervousness"]: 0.30,
                 GoE_IDX["surprise"]: 0.15},
    "joy":      {GoE_IDX["joy"]: 0.40, GoE_IDX["amusement"]: 0.20,
                 GoE_IDX["excitement"]: 0.15, GoE_IDX["approval"]: 0.15,
                 GoE_IDX["optimism"]: 0.10},
    "neutral":  {GoE_IDX["neutral"]: 0.65, GoE_IDX["approval"]: 0.15,
                 GoE_IDX["caring"]: 0.10, GoE_IDX["curiosity"]: 0.10},
    "sadness":  {GoE_IDX["sadness"]: 0.50, GoE_IDX["grief"]: 0.20,
                 GoE_IDX["disappointment"]: 0.20, GoE_IDX["remorse"]: 0.10},
    "surprise": {GoE_IDX["surprise"]: 0.50, GoE_IDX["amusement"]: 0.20,
                 GoE_IDX["curiosity"]: 0.15, GoE_IDX["realization"]: 0.15},
}

# BERT soft distribution: 0.85 for matched label, rest spread evenly
def _bert_vec(meld_emotion: str) -> np.ndarray:
    v = np.full(7, 0.15 / 6, dtype=np.float32)
    if meld_emotion in BERT_IDX:
        v[BERT_IDX[meld_emotion]] = 0.85
    return v

# MELD → VADER synthetic [neg, neu, pos, compound]
MELD_VADER: dict[str, list[float]] = {
    "joy":      [0.00, 0.30, 0.70,  0.65],
    "anger":    [0.60, 0.40, 0.00, -0.60],
    "sadness":  [0.50, 0.50, 0.00, -0.50],
    "fear":     [0.40, 0.60, 0.00, -0.40],
    "disgust":  [0.50, 0.50, 0.00, -0.50],
    "surprise": [0.10, 0.60, 0.30,  0.20],
    "neutral":  [0.00, 1.00, 0.00,  0.00],
}

# MELD emotion → CDM intent state index (0–14, matching trainer/data/synthetic.py)
MELD_TO_INTENT: dict[str, int] = {
    "joy":      1,
    "anger":    7,
    "sadness":  8,
    "fear":     5,
    "disgust":  7,
    "surprise": 3,
    "neutral":  0,
}

# CDM slot indices (matches shared/constants.py CTX_* exactly)
N_CDM_STATES = 15
_S = N_CDM_STATES
CTX_RESIDENCY      = _S        # 15
CTX_TRANSITION     = slice(_S + 1, _S + 4)  # [16:19]
CTX_ABRUPTNESS     = _S + 4   # 19
CTX_COHERENCE      = _S + 5   # 20
CTX_ENTROPY        = _S + 6   # 21
CTX_SPK_DIVERGENCE = _S + 7   # 22
CTX_VELOCITY       = _S + 8   # 23
CTX_ACCELERATION   = _S + 9   # 24
CTX_HIST_POS       = _S + 10  # 25
CTX_HIST_NEU       = _S + 11  # 26
CTX_HIST_NEG       = _S + 12  # 27
CTX_RESONANCE      = _S + 13  # 28
CTX_VOLATILITY     = _S + 14  # 29
CTX_CURR_VALENCE   = _S + 15  # 30
CTX_MSG_LENGTH     = _S + 16  # 31
CTX_LATENCY_MS     = _S + 17  # 32
CTX_HMM_CONF       = _S + 18  # 33
CTX_HMM_ENTROPY    = _S + 19  # 34
CTX_HMM_EMISSION   = _S + 20  # 35
CTX_HMM_NEXT3      = slice(_S + 21, _S + 24)  # [36:39]
CTX_INTENT_STAB    = _S + 24  # 39

CDM_CTX_DIM = 40
MSG_DIM     = 28 + 7 + 4 + CDM_CTX_DIM  # 79


# ── CDM accumulator ───────────────────────────────────────────────────────────

def _build_cdm(
    history: list,          # [{valence, intent_state, speaker}]
    current_emotion: str,
    current_text: str,
) -> np.ndarray:
    """Build 40-dim CDM vector from conversation history — matches meld.py logic."""
    ctx           = np.zeros(CDM_CTX_DIM, dtype=np.float32)
    current_intent = MELD_TO_INTENT.get(current_emotion, 0)
    current_valence = MELD_VADER.get(current_emotion, [0, 1, 0, 0])[3]  # compound
    n = len(history)

    # [0:15] CDM intent one-hot
    ctx[current_intent] = 1.0

    # [15] state_residency
    streak = 1
    for h in reversed(history):
        if h["intent_state"] == current_intent:
            streak += 1
        else:
            break
    ctx[CTX_RESIDENCY] = min(streak / max(n + 1, 1), 1.0)

    # [16:19] transition_path
    recent = [h["intent_state"] for h in history[-3:]]
    for i, s in enumerate(recent):
        ctx[CTX_TRANSITION.start + i] = s / N_CDM_STATES

    # [19] entry_abruptness — valence jump from previous
    prev_val = history[-1]["valence"] if history else current_valence
    ctx[CTX_ABRUPTNESS] = min(abs(current_valence - prev_val), 1.0)

    # [20] topic_coherence
    if n > 0:
        same = sum(1 for h in history[-5:] if h["intent_state"] == current_intent)
        ctx[CTX_COHERENCE] = same / min(n, 5)
    else:
        ctx[CTX_COHERENCE] = 0.5

    # [21] emotion_entropy — diversity of intent states in recent history
    if n > 0:
        ec    = Counter(h["intent_state"] for h in history[-5:])
        total = sum(ec.values())
        probs = [c / total for c in ec.values()]
        ent   = -sum(p * math.log(p + 1e-9) for p in probs)
        ctx[CTX_ENTROPY] = float(min(ent / math.log(N_CDM_STATES), 1.0))

    # [22] speaker_divergence — did speaker change?
    if history:
        ctx[CTX_SPK_DIVERGENCE] = float(history[-1]["speaker"] != "?")

    # [23] velocity
    velocity = current_valence - prev_val
    ctx[CTX_VELOCITY] = float(max(-1.0, min(1.0, velocity)))

    # [24] acceleration
    if len(history) >= 2:
        prev_vel = history[-1]["valence"] - history[-2]["valence"]
        ctx[CTX_ACCELERATION] = float(max(-1.0, min(1.0, velocity - prev_vel)))

    # [25:28] valence history buckets
    all_vals = [h["valence"] for h in history] + [current_valence]
    ctx[CTX_HIST_POS] = float(sum(v >  0.2 for v in all_vals) / len(all_vals))
    ctx[CTX_HIST_NEU] = float(sum(abs(v) <= 0.2 for v in all_vals) / len(all_vals))
    ctx[CTX_HIST_NEG] = float(sum(v < -0.2 for v in all_vals) / len(all_vals))

    # [28] topic_resonance — reuse coherence
    ctx[CTX_RESONANCE] = ctx[CTX_COHERENCE]

    # [29] volatility
    if len(all_vals) > 1:
        ctx[CTX_VOLATILITY] = float(min(float(np.std(all_vals)), 1.0))

    # [30] current_valence
    ctx[CTX_CURR_VALENCE] = float(max(-1.0, min(1.0, current_valence)))

    # [31] message_length (chars, normalized)
    ctx[CTX_MSG_LENGTH] = min(len(current_text) / 200.0, 1.0)

    # [32] latency — not available in MELD
    ctx[CTX_LATENCY_MS] = 0.0

    # [33:36] HMM features — moderate synthetic values
    ctx[CTX_HMM_CONF]     = 0.65
    ctx[CTX_HMM_ENTROPY]  = 0.40
    ctx[CTX_HMM_EMISSION] = 0.65
    ctx[CTX_HMM_NEXT3.start]     = 0.50
    ctx[CTX_HMM_NEXT3.start + 1] = 0.30
    ctx[CTX_HMM_NEXT3.start + 2] = 0.20

    # [39] intent_stability
    ctx[CTX_INTENT_STAB] = ctx[CTX_RESIDENCY]

    assert ctx.shape[0] == CDM_CTX_DIM
    return ctx


def _goe_vec(meld_emotion: str) -> np.ndarray:
    v = np.zeros(28, dtype=np.float32)
    for idx, val in MELD_TO_GoE.get(meld_emotion, {GoE_IDX["neutral"]: 1.0}).items():
        v[idx] = val
    return v


def _build_feature(emotion: str, text: str, history: list) -> np.ndarray:
    goe   = _goe_vec(emotion)
    bert  = _bert_vec(emotion)
    vader = np.array(MELD_VADER.get(emotion, [0.0, 1.0, 0.0, 0.0]), dtype=np.float32)

    if history:
        cdm = _build_cdm(history, emotion, text)
    else:
        # First utterance: minimal CDM
        cdm = np.zeros(CDM_CTX_DIM, dtype=np.float32)
        intent_idx = MELD_TO_INTENT.get(emotion, 0)
        cdm[intent_idx] = 1.0
        compound = MELD_VADER.get(emotion, [0, 1, 0, 0])[3]
        cdm[CTX_CURR_VALENCE] = compound
        cdm[CTX_MSG_LENGTH]   = min(len(text) / 200.0, 1.0)
        cdm[CTX_HMM_CONF]     = 0.30
        cdm[CTX_HMM_NEXT3.start]     = 0.50
        cdm[CTX_HMM_NEXT3.start + 1] = 0.30
        cdm[CTX_HMM_NEXT3.start + 2] = 0.20

    feat = np.concatenate([goe, bert, vader, cdm])
    assert feat.shape[0] == MSG_DIM, f"Feature dim {feat.shape[0]} ≠ {MSG_DIM}"
    return feat


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sequences() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (X, Y) where X[t] is features, Y[t] is GoE target for t+1."""
    with open(MELD_PATH) as f:
        raw = json.load(f)

    by_dialog: dict[str, list] = defaultdict(list)
    for row in raw:
        by_dialog[row["d"]].append(row)

    sequences = []
    for utts in by_dialog.values():
        utts = sorted(utts, key=lambda r: r["u"])
        if len(utts) < 2:
            continue

        history: list[dict] = []
        feats, targets = [], []

        for utt in utts:
            emo  = utt["e"]
            text = utt["t"]
            feat = _build_feature(emo, text, history)
            goe  = _goe_vec(emo)
            feats.append(feat)
            targets.append(goe)
            history.append({
                "valence":      MELD_VADER.get(emo, [0, 1, 0, 0])[3],
                "intent_state": MELD_TO_INTENT.get(emo, 0),
                "speaker":      utt.get("s", "?"),
            })

        # X[0..N-2] predicts Y[1..N-1]
        X = np.stack(feats[:-1])    # [N-1, 79]
        Y = np.stack(targets[1:])   # [N-1, 28]
        sequences.append((X, Y))

    print(f"Loaded {len(sequences)} sequences, "
          f"{sum(len(X) for X, _ in sequences)} step pairs total.")
    return sequences


# ── Training ──────────────────────────────────────────────────────────────────

def topk_accuracy(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """Top-k accuracy where target class = argmax of soft target distribution."""
    target_class = target.argmax(dim=-1)          # [B]
    topk_preds   = pred.topk(k, dim=-1).indices   # [B, k]
    correct = (topk_preds == target_class.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def _weighted_loss(pred: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted KL divergence with soft targets.
    pred outputs Softmax probabilities (sums to 1), same as soft targets.

    KL = sum_c target_c * log(target_c / pred_c)
       = -sum_c target_c * log(pred_c)  + const (ignored for optimization)

    We use the cross-entropy term and weight each step by its dominant class weight.
    """
    ce           = -(target * torch.log(pred.clamp(min=1e-8))).sum(dim=-1)  # [B, T]
    tgt_class    = target.argmax(dim=-1)                                     # [B, T]
    step_weight  = class_weights[tgt_class]                                  # [B, T]
    return (ce * step_weight).mean()


def train(sequences: list, device: torch.device, epochs: int, lr: float, hidden: int):
    random.shuffle(sequences)
    n_val    = max(1, int(len(sequences) * 0.12))
    val_seqs = sequences[:n_val]
    trn_seqs = sequences[n_val:]
    print(f"Train: {len(trn_seqs)} convs | Val: {len(val_seqs)} convs | device: {device}")

    # Class weights — inverse frequency to suppress neutral dominance
    all_targets = np.concatenate([Y for _, Y in trn_seqs], axis=0)   # [N, 28]
    class_counts = all_targets.argmax(axis=1)                         # dominant class
    freq = np.bincount(class_counts, minlength=28).astype(np.float32)
    freq = np.where(freq == 0, 1.0, freq)
    weights = (freq.sum() / (28 * freq))                              # inv-freq
    weights = np.clip(weights, 0.3, 4.0)                              # cap extreme weights
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = ConversationLSTM(
        input_dim=MSG_DIM, hidden_dim=hidden, num_layers=2,
        output_dim=28, dropout=0.2,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.05,
    )

    best_val   = float("inf")
    best_top1  = 0.0
    best_top3  = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(trn_seqs)
        trn_loss = trn_steps = 0

        for X, Y in trn_seqs:
            x = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, 79]
            y = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, 28]
            opt.zero_grad()
            pred, _ = model(x)
            loss = _weighted_loss(pred, y, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trn_loss  += loss.item() * X.shape[0]
            trn_steps += X.shape[0]

        scheduler.step()

        # Validation — measure top-k accuracy (track best by top3 acc)
        model.eval()
        val_loss = val_steps = 0
        all_pred, all_tgt = [], []

        with torch.no_grad():
            for X, Y in val_seqs:
                x = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
                y = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(0)
                pred, _ = model(x)
                val_loss  += _weighted_loss(pred, y, class_weights).item() * X.shape[0]
                val_steps += X.shape[0]
                all_pred.append(pred.squeeze(0))
                all_tgt.append(y.squeeze(0))

        p_cat = torch.cat(all_pred, dim=0)
        t_cat = torch.cat(all_tgt,  dim=0)
        top1  = topk_accuracy(p_cat, t_cat, 1)
        top3  = topk_accuracy(p_cat, t_cat, 3)
        avg_val = val_loss / max(val_steps, 1)

        if top3 > best_top3 or (top3 == best_top3 and top1 > best_top1):
            best_val  = avg_val
            best_top1 = top1
            best_top3 = top3
            torch.save(model.state_dict(), MODEL_OUT)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"trn={trn_loss/trn_steps:.3f}  "
                  f"val={avg_val:.3f}  "
                  f"top1={top1:.3f}  top3={top3:.3f}")

    return best_top1, best_top3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # MPS crashes during LSTM backward with cosine scheduler — use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sequences = load_sequences()
    top1, top3 = train(sequences, device, args.epochs, args.lr, args.hidden)

    config = {
        "input_dim":  MSG_DIM,
        "hidden_dim": args.hidden,
        "num_layers": 2,
        "output_dim": 28,
        "dropout":    0.2,
        "n_train":    len(sequences),
        "top1_acc":   round(top1, 4),
        "top3_acc":   round(top3, 4),
        "trained_on": "MELD-9989",
    }
    with open(CONFIG_OUT, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone — best val top1={top1:.3f} top3={top3:.3f}")
    print(f"Saved: {MODEL_OUT}")
    print(f"       {CONFIG_OUT}")


if __name__ == "__main__":
    main()
