#!/usr/bin/env python3
"""
trajectory_train.py — Stage 3+: Retrain ConversationLSTM on MELD + EmpatheticDialogues.

Data sources:
  MELD (968 convs)                  — .cache/meld_raw_cache.json (pre-cached)
  EmpatheticDialogues (17,844 convs) — facebook/empathetic_dialogues via HuggingFace

Feature vector per utterance: 79 dims
  [  0: 27]  go_emotions    28 dims — soft distribution from emotion label
  [ 28: 34]  basic_bert      7 dims — soft distribution from emotion label
  [ 35: 38]  vader           4 dims — synthetic valence features
  [ 39: 78]  cdm_context    40 dims — built from real conversation history

Output: .cache/trajectory_lstm.pt + .cache/trajectory_config.json

Run:
  python trajectory_train.py [--epochs N] [--lr F] [--hidden N]
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ROOT  = Path(__file__).parent
CACHE = ROOT / ".cache"
sys.path.insert(0, str(ROOT / "central_responder_service"))
from trajectory.model import ConversationLSTM

MELD_PATH   = CACHE / "meld_raw_cache.json"
EMPAT_CACHE = CACHE / "empathetic_raw_cache.json"
MODEL_OUT   = CACHE / "trajectory_lstm.pt"
CONFIG_OUT  = CACHE / "trajectory_config.json"

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

# GoEmotions 28 → BERT 7 closest match (for bert_vec building)
GoE_TO_BERT = {
    "admiration": "joy",      "amusement": "joy",      "anger": "anger",
    "annoyance": "anger",     "approval": "joy",       "caring": "joy",
    "confusion": "neutral",   "curiosity": "neutral",  "desire": "joy",
    "disappointment": "sadness", "disapproval": "anger","disgust": "disgust",
    "embarrassment": "sadness","excitement": "joy",    "fear": "fear",
    "gratitude": "joy",       "grief": "sadness",      "joy": "joy",
    "love": "joy",            "nervousness": "fear",   "optimism": "joy",
    "pride": "joy",           "realization": "neutral","relief": "joy",
    "remorse": "sadness",     "sadness": "sadness",    "surprise": "surprise",
    "neutral": "neutral",
}

# GoEmotions 28 → synthetic valence [neg, neu, pos, compound]
GoE_VADER = {
    "admiration":   [0.00, 0.20, 0.80,  0.75],
    "amusement":    [0.00, 0.30, 0.70,  0.60],
    "anger":        [0.70, 0.30, 0.00, -0.70],
    "annoyance":    [0.50, 0.50, 0.00, -0.40],
    "approval":     [0.00, 0.20, 0.80,  0.50],
    "caring":       [0.00, 0.30, 0.70,  0.55],
    "confusion":    [0.10, 0.80, 0.10, -0.10],
    "curiosity":    [0.00, 0.50, 0.50,  0.20],
    "desire":       [0.00, 0.30, 0.70,  0.40],
    "disappointment":[0.50,0.50, 0.00, -0.50],
    "disapproval":  [0.50, 0.50, 0.00, -0.50],
    "disgust":      [0.60, 0.40, 0.00, -0.60],
    "embarrassment":[0.40, 0.60, 0.00, -0.30],
    "excitement":   [0.00, 0.20, 0.80,  0.70],
    "fear":         [0.50, 0.50, 0.00, -0.55],
    "gratitude":    [0.00, 0.10, 0.90,  0.80],
    "grief":        [0.70, 0.30, 0.00, -0.75],
    "joy":          [0.00, 0.20, 0.80,  0.70],
    "love":         [0.00, 0.10, 0.90,  0.85],
    "nervousness":  [0.40, 0.60, 0.00, -0.35],
    "optimism":     [0.00, 0.20, 0.80,  0.60],
    "pride":        [0.00, 0.20, 0.80,  0.65],
    "realization":  [0.00, 0.70, 0.30,  0.10],
    "relief":       [0.00, 0.30, 0.70,  0.50],
    "remorse":      [0.50, 0.50, 0.00, -0.45],
    "sadness":      [0.60, 0.40, 0.00, -0.55],
    "surprise":     [0.05, 0.55, 0.40,  0.15],
    "neutral":      [0.00, 1.00, 0.00,  0.00],
}

# GoEmotions → CDM intent state index (0-14, matching shared/constants.py CDM_STATES)
GoE_INTENT = {
    "admiration": 2,  "amusement": 4,   "anger": 7,       "annoyance": 6,
    "approval": 0,    "caring": 12,     "confusion": 8,   "curiosity": 10,
    "desire": 9,      "disappointment":3,"disapproval": 5, "disgust": 5,
    "embarrassment":3,"excitement": 2,  "fear": 2,        "gratitude": 14,
    "grief": 3,       "joy": 1,         "love": 1,        "nervousness": 2,
    "optimism": 4,    "pride": 2,       "realization": 8, "relief": 4,
    "remorse": 3,     "sadness": 3,     "surprise": 1,    "neutral": 0,
}

# MELD 7-class → GoEmotions 28 sharper soft distributions (primary >= 0.60)
MELD_TO_GoE: dict[str, dict[int, float]] = {
    "anger":    {GoE_IDX["anger"]: 0.60, GoE_IDX["annoyance"]: 0.20,
                 GoE_IDX["disapproval"]: 0.12, GoE_IDX["disgust"]: 0.08},
    "disgust":  {GoE_IDX["disgust"]: 0.65, GoE_IDX["disapproval"]: 0.20,
                 GoE_IDX["annoyance"]: 0.15},
    "fear":     {GoE_IDX["fear"]: 0.60, GoE_IDX["nervousness"]: 0.28,
                 GoE_IDX["surprise"]: 0.12},
    "joy":      {GoE_IDX["joy"]: 0.55, GoE_IDX["amusement"]: 0.20,
                 GoE_IDX["excitement"]: 0.13, GoE_IDX["approval"]: 0.07,
                 GoE_IDX["optimism"]: 0.05},
    "neutral":  {GoE_IDX["neutral"]: 0.70, GoE_IDX["approval"]: 0.12,
                 GoE_IDX["curiosity"]: 0.10, GoE_IDX["caring"]: 0.08},
    "sadness":  {GoE_IDX["sadness"]: 0.55, GoE_IDX["grief"]: 0.20,
                 GoE_IDX["disappointment"]: 0.15, GoE_IDX["remorse"]: 0.10},
    "surprise": {GoE_IDX["surprise"]: 0.55, GoE_IDX["amusement"]: 0.20,
                 GoE_IDX["curiosity"]: 0.15, GoE_IDX["realization"]: 0.10},
}

# EmpatheticDialogues 32 emotions → GoEmotions 28 (via trainer/data/empathetic.py)
EMPATHETIC_TO_GoE_LABEL: dict[str, str] = {
    "sentimental": "love",       "afraid": "fear",         "proud": "pride",
    "faithful": "caring",        "terrified": "fear",      "joyful": "joy",
    "angry": "anger",            "sad": "sadness",         "jealous": "disapproval",
    "grateful": "gratitude",     "embarrassed": "embarrassment",
    "excited": "excitement",     "annoyed": "annoyance",   "lonely": "sadness",
    "surprised": "surprise",     "furious": "anger",       "disappointed": "disappointment",
    "caring": "caring",          "trusting": "approval",   "disgusted": "disgust",
    "anticipating": "desire",    "anxious": "nervousness", "nostalgic": "curiosity",
    "confident": "pride",        "devastated": "grief",    "hopeful": "optimism",
    "guilty": "remorse",         "impressed": "admiration","apprehensive": "nervousness",
    "touched": "caring",
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
CTX_LATENCY_MS     = _S + 17  # 32  — repurposed as normalized turn index
CTX_HMM_CONF       = _S + 18  # 33
CTX_HMM_ENTROPY    = _S + 19  # 34
CTX_HMM_EMISSION   = _S + 20  # 35
CTX_HMM_NEXT3      = slice(_S + 21, _S + 24)  # [36:39]
CTX_INTENT_STAB    = _S + 24  # 39

CDM_CTX_DIM = 40
MSG_DIM     = 28 + 7 + 4 + CDM_CTX_DIM  # 79


# ── Feature builders ─────────────────────────────────────────────────────────

def _goe_vec(goe_label: str) -> np.ndarray:
    """28-dim probability distribution for a GoEmotions label."""
    v = np.zeros(28, dtype=np.float32)
    if goe_label in GoE_IDX:
        # single-label: use GoE_VADER compound to build a soft distribution
        # but for MELD this is called via MELD_TO_GoE, so this handles
        # EmpatheticDialogues labels (already in GoE label space)
        v[GoE_IDX[goe_label]] = 0.75
        # add small mass to related emotions via GoE_VADER proximity
        compound = GoE_VADER.get(goe_label, [0, 1, 0, 0])[3]
        if compound > 0.5:
            for related in ["approval", "optimism"]:
                if related != goe_label and related in GoE_IDX:
                    v[GoE_IDX[related]] += 0.08
        elif compound < -0.4:
            for related in ["disapproval", "disappointment"]:
                if related != goe_label and related in GoE_IDX:
                    v[GoE_IDX[related]] += 0.06
        v /= v.sum()
    else:
        v[GoE_IDX["neutral"]] = 1.0
    return v


def _goe_vec_meld(meld_emotion: str) -> np.ndarray:
    """28-dim soft distribution from a MELD 7-class emotion label."""
    v = np.zeros(28, dtype=np.float32)
    for idx, val in MELD_TO_GoE.get(meld_emotion, {GoE_IDX["neutral"]: 1.0}).items():
        v[idx] = val
    return v


def _bert_vec(goe_label: str) -> np.ndarray:
    """7-dim soft distribution: 0.85 for matched BERT label, rest spread evenly."""
    bert_label = GoE_TO_BERT.get(goe_label, "neutral")
    v = np.full(7, 0.15 / 6, dtype=np.float32)
    v[BERT_IDX[bert_label]] = 0.85
    return v


def _build_cdm(
    history: list,
    current_emotion: str,
    current_text: str,
    current_speaker: str = "?",
) -> np.ndarray:
    """
    Build 40-dim CDM context vector from conversation history.
    Matches slot layout of shared/constants.py CTX_* constants.
    """
    ctx             = np.zeros(CDM_CTX_DIM, dtype=np.float32)
    current_intent  = GoE_INTENT.get(current_emotion, 0)
    compound        = GoE_VADER.get(current_emotion, [0, 1, 0, 0])[3]
    n               = len(history)

    # [0:15] CDM intent one-hot
    ctx[current_intent] = 1.0

    # [15] state_residency — consecutive same intent streak
    streak = 1
    for h in reversed(history):
        if h["intent_state"] == current_intent:
            streak += 1
        else:
            break
    ctx[CTX_RESIDENCY] = min(streak / max(n + 1, 1), 1.0)

    # [16:19] transition_path — last 3 intent indices / N_CDM_STATES
    recent = [h["intent_state"] for h in history[-3:]]
    for i, s in enumerate(recent):
        ctx[CTX_TRANSITION.start + i] = s / N_CDM_STATES

    # [19] entry_abruptness — valence jump from previous
    prev_val = history[-1]["valence"] if history else compound
    ctx[CTX_ABRUPTNESS] = min(abs(compound - prev_val), 1.0)

    # [20] topic_coherence — stable intent = high coherence
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
        ctx[CTX_ENTROPY] = float(min(ent / math.log(N_CDM_STATES + 1e-9), 1.0))

    # [22] speaker_divergence — did speaker change from PREVIOUS utterance?
    if history:
        prev_speaker = history[-1].get("speaker", "?")
        ctx[CTX_SPK_DIVERGENCE] = float(prev_speaker != current_speaker)

    # [23] velocity — Δcompound from previous
    velocity = compound - prev_val
    ctx[CTX_VELOCITY] = float(max(-1.0, min(1.0, velocity)))

    # [24] acceleration — Δ²compound
    if len(history) >= 2:
        prev_vel = history[-1]["valence"] - history[-2]["valence"]
        ctx[CTX_ACCELERATION] = float(max(-1.0, min(1.0, velocity - prev_vel)))

    # [25:28] valence history buckets
    all_vals = [h["valence"] for h in history] + [compound]
    ctx[CTX_HIST_POS] = float(sum(v >  0.2 for v in all_vals) / len(all_vals))
    ctx[CTX_HIST_NEU] = float(sum(abs(v) <= 0.2 for v in all_vals) / len(all_vals))
    ctx[CTX_HIST_NEG] = float(sum(v < -0.2 for v in all_vals) / len(all_vals))

    # [28] topic_resonance — reuse coherence
    ctx[CTX_RESONANCE] = ctx[CTX_COHERENCE]

    # [29] volatility — std of recent valence window
    if len(all_vals) > 1:
        ctx[CTX_VOLATILITY] = float(min(float(np.std(all_vals)), 1.0))

    # [30] current_valence
    ctx[CTX_CURR_VALENCE] = float(max(-1.0, min(1.0, compound)))

    # [31] message_length (normalized)
    ctx[CTX_MSG_LENGTH] = min(len(current_text) / 200.0, 1.0)

    # [32] turn index — normalized position within conversation (was always 0)
    ctx[CTX_LATENCY_MS] = min(n / 20.0, 1.0)

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


def _build_feature(
    goe_label: str,
    text: str,
    history: list,
    current_speaker: str = "?",
    meld_label: Optional[str] = None,
) -> np.ndarray:
    """Build 79-dim feature vector for one utterance."""
    goe   = _goe_vec_meld(meld_label) if meld_label else _goe_vec(goe_label)
    bert  = _bert_vec(goe_label)
    vader = np.array(GoE_VADER.get(goe_label, [0.0, 1.0, 0.0, 0.0]), dtype=np.float32)

    if history:
        cdm = _build_cdm(history, goe_label, text, current_speaker)
    else:
        cdm = np.zeros(CDM_CTX_DIM, dtype=np.float32)
        cdm[GoE_INTENT.get(goe_label, 0)] = 1.0
        cdm[CTX_CURR_VALENCE] = float(max(-1.0, min(1.0, vader[3])))
        cdm[CTX_MSG_LENGTH]   = min(len(text) / 200.0, 1.0)
        cdm[CTX_HMM_CONF]     = 0.30
        cdm[CTX_HMM_NEXT3.start]     = 0.50
        cdm[CTX_HMM_NEXT3.start + 1] = 0.30
        cdm[CTX_HMM_NEXT3.start + 2] = 0.20

    feat = np.concatenate([goe, bert, vader, cdm])
    assert feat.shape[0] == MSG_DIM, f"Feature dim {feat.shape[0]} ≠ {MSG_DIM}"
    return feat


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_meld_sequences() -> list:
    """Load MELD (968 multi-emotion conversations, ~8,951 step pairs)."""
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
            meld_emo = utt["e"]
            text     = utt["t"]
            speaker  = utt.get("s", "?")
            # Map MELD→GoE for CDM/BERT/VADER — use primary GoE label
            primary_goe = max(MELD_TO_GoE.get(meld_emo, {GoE_IDX["neutral"]: 1.0}),
                              key=MELD_TO_GoE.get(meld_emo, {GoE_IDX["neutral"]: 1.0}).get)
            primary_goe_label = EMOTION_LABELS_28[primary_goe]

            feat   = _build_feature(primary_goe_label, text, history, speaker, meld_label=meld_emo)
            target = _goe_vec_meld(meld_emo)   # target is MELD→GoE soft distribution
            feats.append(feat)
            targets.append(target)
            history.append({
                "valence":      GoE_VADER.get(primary_goe_label, [0, 1, 0, 0])[3],
                "intent_state": GoE_INTENT.get(primary_goe_label, 0),
                "speaker":      speaker,
            })

        X = np.stack(feats[:-1])
        Y = np.stack(targets[1:])
        sequences.append((X, Y))

    print(f"MELD: {len(sequences)} sequences, {sum(len(X) for X, _ in sequences)} step pairs")
    return sequences


def load_empathetic_sequences(max_convs: int = 8000) -> list:
    """
    Load EmpatheticDialogues (up to max_convs conversations).
    Caches to .cache/empathetic_raw_cache.json after first download.
    """
    # Load from cache or download
    if EMPAT_CACHE.exists():
        with open(EMPAT_CACHE) as f:
            raw = json.load(f)
        print(f"EmpatheticDialogues: loaded {len(raw)} rows from cache")
    else:
        print("EmpatheticDialogues: downloading from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("facebook/empathetic_dialogues", split="train")
        raw = [
            {
                "conv_id":       r["conv_id"],
                "utt_idx":       r["utterance_idx"],
                "context":       r["context"],
                "utterance":     r["utterance"].replace("_comma_", ",").strip(),
                "speaker_idx":   r["speaker_idx"],
            }
            for r in ds
        ]
        with open(EMPAT_CACHE, "w") as f:
            json.dump(raw, f)
        print(f"EmpatheticDialogues: cached {len(raw)} rows → {EMPAT_CACHE}")

    by_conv: dict[str, list] = defaultdict(list)
    for row in raw:
        by_conv[row["conv_id"]].append(row)

    # Limit to max_convs
    conv_ids = list(by_conv.keys())
    random.shuffle(conv_ids)
    conv_ids = conv_ids[:max_convs]

    sequences = []
    for conv_id in conv_ids:
        utts = sorted(by_conv[conv_id], key=lambda r: r["utt_idx"])
        if len(utts) < 2:
            continue

        # Map empathetic emotion label → GoEmotions 28
        context_emo = utts[0]["context"].lower()
        goe_label   = EMPATHETIC_TO_GoE_LABEL.get(context_emo, "neutral")

        # Responder (speaker_idx=0) gets "caring" emotion (empathetic listening)
        SPEAKER_EMO = {
            "0": "caring",   # responder / empathetic listener
            "1": goe_label,  # story-teller carries the conversation emotion
        }

        history: list[dict] = []
        feats, targets = [], []

        for utt in utts:
            spk      = str(utt["speaker_idx"])
            emo      = SPEAKER_EMO.get(spk, goe_label)
            text     = utt["utterance"]
            feat     = _build_feature(emo, text, history, spk)
            target   = _goe_vec(emo)
            feats.append(feat)
            targets.append(target)
            history.append({
                "valence":      GoE_VADER.get(emo, [0, 1, 0, 0])[3],
                "intent_state": GoE_INTENT.get(emo, 0),
                "speaker":      spk,
            })

        X = np.stack(feats[:-1])
        Y = np.stack(targets[1:])
        sequences.append((X, Y))

    total_pairs = sum(len(X) for X, _ in sequences)
    print(f"EmpatheticDialogues: {len(sequences)} sequences, {total_pairs} step pairs")
    return sequences


# ── Training ──────────────────────────────────────────────────────────────────

def topk_accuracy(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
    target_class = target.argmax(dim=-1)
    topk_preds   = pred.topk(k, dim=-1).indices
    correct = (topk_preds == target_class.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def _weighted_kl(pred: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted cross-entropy (KL divergence w/ soft targets).
    pred: [B, T, 28] Softmax probabilities (sums to 1 per step)
    target: [B, T, 28] soft target distribution
    """
    ce          = -(target * torch.log(pred.clamp(min=1e-8))).sum(dim=-1)  # [B, T]
    tgt_class   = target.argmax(dim=-1)
    step_weight = class_weights[tgt_class]
    return (ce * step_weight).mean()


def train(
    sequences: list,
    device: torch.device,
    epochs: int,
    lr: float,
    hidden: int,
):
    random.shuffle(sequences)
    n_val    = max(1, int(len(sequences) * 0.10))
    val_seqs = sequences[:n_val]
    trn_seqs = sequences[n_val:]
    print(f"Train: {len(trn_seqs)} | Val: {len(val_seqs)} | device: {device} | hidden: {hidden}")

    # Class weights — inverse frequency capped to suppress neutral dominance
    all_targets  = np.concatenate([Y for _, Y in trn_seqs], axis=0)
    class_counts = all_targets.argmax(axis=1)
    freq         = np.bincount(class_counts, minlength=28).astype(np.float32)
    freq         = np.where(freq == 0, 1.0, freq)
    weights      = freq.sum() / (28 * freq)
    weights      = np.clip(weights, 0.25, 5.0)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = ConversationLSTM(
        input_dim=MSG_DIM, hidden_dim=hidden, num_layers=2,
        output_dim=28, dropout=0.25,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(epochs // 3, 15), T_mult=1, eta_min=lr * 0.02,
    )

    best_top3 = 0.0
    best_top1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(trn_seqs)
        trn_loss = trn_steps = 0

        for X, Y in trn_seqs:
            x = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(0)
            opt.zero_grad()
            pred, _ = model(x)
            loss = _weighted_kl(pred, y, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trn_loss  += loss.item() * X.shape[0]
            trn_steps += X.shape[0]

        scheduler.step()

        model.eval()
        all_pred, all_tgt = [], []
        val_loss = val_steps = 0

        with torch.no_grad():
            for X, Y in val_seqs:
                x = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
                y = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(0)
                pred, _ = model(x)
                val_loss  += _weighted_kl(pred, y, class_weights).item() * X.shape[0]
                val_steps += X.shape[0]
                all_pred.append(pred.squeeze(0))
                all_tgt.append(y.squeeze(0))

        p_cat = torch.cat(all_pred, dim=0)
        t_cat = torch.cat(all_tgt,  dim=0)
        top1  = topk_accuracy(p_cat, t_cat, 1)
        top3  = topk_accuracy(p_cat, t_cat, 3)

        if top3 > best_top3 or (top3 == best_top3 and top1 > best_top1):
            best_top3 = top3
            best_top1 = top1
            torch.save(model.state_dict(), MODEL_OUT)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"trn={trn_loss/trn_steps:.3f}  "
                  f"val={val_loss/val_steps:.3f}  "
                  f"top1={top1:.3f}  top3={top3:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")

    return best_top1, best_top3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=80)
    parser.add_argument("--lr",        type=float, default=8e-4)
    parser.add_argument("--hidden",    type=int,   default=256)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--empat-max", type=int,   default=8000,
                        dest="empat_max",
                        help="Max EmpatheticDialogues conversations to use")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    meld_seqs   = load_meld_sequences()
    empat_seqs  = load_empathetic_sequences(max_convs=args.empat_max)
    sequences   = meld_seqs + empat_seqs
    random.shuffle(sequences)

    total_pairs = sum(len(X) for X, _ in sequences)
    print(f"\nTotal: {len(sequences)} sequences, {total_pairs} step pairs")

    top1, top3 = train(sequences, device, args.epochs, args.lr, args.hidden)

    config = {
        "input_dim":  MSG_DIM,
        "hidden_dim": args.hidden,
        "num_layers": 2,
        "output_dim": 28,
        "dropout":    0.25,
        "n_train":    len(sequences),
        "top1_acc":   round(top1, 4),
        "top3_acc":   round(top3, 4),
        "trained_on": f"MELD+EmpatheticDialogues",
    }
    with open(CONFIG_OUT, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone — best top1={top1:.3f}  top3={top3:.3f}")
    print(f"Saved: {MODEL_OUT}")
    print(f"       {CONFIG_OUT}")


if __name__ == "__main__":
    main()
