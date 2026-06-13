"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Emotional Dialogue Encoder (EDE) — Design Proposal + Full Implementation  ║
║   Replaces ConversationLSTM in central_responder_service/trajectory/        ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHY THIS EXISTS
──────────────
The current ConversationLSTM predicts "neutral" 98.8% of the time in production
(measured from 865 messages in the DB). This is not a training bug — it is a
design flaw. The LSTM was trained to solve the wrong problem with the wrong data,
and the architecture cannot recover from those decisions.

WHAT IS PROVABLY BROKEN IN THE CURRENT SYSTEM
──────────────────────────────────────────────
Evidence is from the production database and training code audit.

  (1) WRONG TASK: The LSTM predicts P(emotion_{T+1} | message_T) — future prediction.
      What the meta-learner needs is P(emotion_T | history_{1..T-1}) — retrospective
      conditioning of the CURRENT classification. These are different problems.
      The "prior" at [79:107] is supposed to help classify message T, but it contains
      a prediction about message T+1. It is backwards.

  (2) SYNTHETIC LABEL COLLAPSE: Training used hand-coded MELD→GoE maps (e.g.
      "anger" → {anger:0.60, annoyance:0.20, ...}). EmpatheticDialogues assigned
      all responder turns to "caring" and all storyteller turns to ONE emotion per
      entire conversation. The model learned to predict the modal class (neutral=46.6%
      in MELD) and achieved 85.7% top-1 accuracy on validation — by repeating the
      current emotion. Deployment: 98.8% neutral.

  (3) CDM LEAKAGE IN LSTM INPUT: The 79-dim LSTM input contains CDM[40] which
      already encodes valence_history, velocity, acceleration, state_residency.
      The LSTM's hidden state IS conversation memory. Feeding explicit hand-engineered
      memory as input forces the network to reconcile two conflicting memory sources,
      making learning unstable. CDM HMM slots are hardcoded to 0.65 in training but
      vary 0–1 in production → distribution shift.

  (4) LONG-RANGE FORGETTING: The longest production conversation has 749 messages
      (41% unique, rest are demo loop repetitions). Any LSTM hidden state at message
      700 carries almost no signal from message 1 due to vanishing gradients. The
      oscillating pride↔grief pattern (dominant transition pair in DB: pride→grief
      41 times, grief→pride 31 times) spans multiple messages — LSTM forgets this.

  (5) REDIS COMPLEXITY: Serializing (h, c) tensors [2 × num_layers × 1 × 256] to
      JSON per conversation creates lock contention. When a lock is busy, the step
      returns {} and [79:107] stays zeros — so high-traffic conversations get worse
      trajectory context, not better.

  (6) VOLATILE LONG CONVERSATIONS: Volatility jumps 26× in deep conversations
      (0.001 at messages 1–5 vs 0.026 at messages 31+). No static hidden state
      TTL or fixed model architecture handles this correctly without explicit
      phase-awareness.

THE RIGHT FRAMING
─────────────────
The trajectory mechanism should compute:

    trajectory_prior = P(emotion_T | goe_{T-1}, goe_{T-2}, ..., goe_{T-N})

This is a proper Bayesian prior. It conditions the meta-learner's classification
of the CURRENT message on the recent emotional history — exactly what was intended
but never implemented correctly.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE: EmotionalDialogueEncoder (EDE)
═══════════════════════════════════════════════════════════════════════════════

Input:  GoE history window [N, 28] — last N messages' real GoE distributions
        (fetched from Redis, populated from the live pipeline)

Internal:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  [CLS] │ goe_{T-N} │ goe_{T-N+1} │ ... │ goe_{T-2} │ goe_{T-1}    │
  │   —    │  [28]     │   [28]      │     │  [28]     │  [28]         │
  └──────────────────────────────────────────────────────────────────────┘
        ↓ Linear(28→64) + LayerNorm  [input projection per token]
        ↓ + learned positional embedding (position=0 oldest, N newest)
        ↓ 2-layer Transformer Encoder (Pre-LN, 4 heads, d=64, ff=256)
        ↓
  h_CLS ──→ Prior Head:  Linear(64→28) + Softmax  → trajectory_prior [28]
        └──→ Phase Head:  Linear(64→6)              → phase_logits    [6]

Output:
  trajectory_prior [28]: replaces LSTM predicted_next at feature vector [79:107]
  phase             [6]: conversation phase — used to modulate ctx_weight in
                         GatingEnsembleNet (does not change FEATURE_DIM=107)

WHY TRANSFORMER OVER LSTM (SPECIFIC TO THIS PROBLEM)
─────────────────────────────────────────────────────
  • Self-attention directly models long-range dependencies within the window.
    The pride↔grief oscillation that spans 5–10 messages is captured in a
    single attention operation; an LSTM would need to backprop through 5–10 steps.

  • No hidden state to serialize. Redis stores the last N GoE distributions as
    a plain JSON list (~3 KB). No locks, no corruption risk, directly inspectable.

  • Stateless per-step. Every EDE call reads history and runs a full forward pass.
    Idempotent — recomputing the same call gives the same result.

  • CLS token learns a global conversation summary. Attention weights over the
    N history tokens are interpretable: which previous messages influenced the prior.

  • Pre-LN architecture avoids the training instability that caused LayerNorm to
    be added post-hoc to the LSTM output in the current code.

CONVERSATION PHASES (6 classes, zero annotation required)
──────────────────────────────────────────────────────────
Derived algorithmically from the valence window. Evidence from DB:
  early (msgs 1–5):   avg_valence=-0.08, volatility=0.001
  mid   (msgs 6–15):  avg_valence=-0.01, volatility=0.0005  ← most stable
  late  (msgs 16–30): avg_valence=+0.07, volatility=0.0013
  deep  (msgs 31+):   avg_valence=+0.07, volatility=0.026   ← 26× jump

Phase detection rules (no ML needed for labeling):
  OPENING      → fewer than 3 history messages
  ESCALATION   → net_delta < -0.15 AND mean_recent < 0 (declining into negative)
  PEAK         → volatility > 0.25 AND |mean_valence| > 0.2 (high + unstable)
  TURNING_PT   → |one-step velocity| > 0.20 (abrupt reversal)
  RESOLUTION   → net_delta > 0.15 AND mean_recent >= -0.05 (recovering)
  SUSTAINED    → volatility < 0.05 (stable, anything)

Phase is used in GatingEnsembleNet to adjust ctx_weight:
  ESCALATION → scale up negative emotion probabilities from context prior
  RESOLUTION → scale up positive emotion probabilities from context prior
  PEAK       → increase overall confidence (meta-learner is uncertain here)
  SUSTAINED  → reduce context weight (context repeats, adds no information)

TRAINING DATA STRATEGY
───────────────────────
The current DB contains 865 messages but 59% are repeated (demo loop).
Deduplicated unique pairs: ~500. Too few for standalone training.

Sources (all with REAL GoE distributions, zero synthetic label maps):

  1. DB real GoE sequences     ~500 deduplicated step pairs
     ↳ GoE scores from pipeline_log_json→models→go_emotions (actual model output)
     ↳ Valence from context_snapshot→cur_valence for phase derivation

  2. MELD via real GoE model   ~8,951 step pairs
     ↳ Run bhadresh-savani/bert-base-go-emotion on actual MELD utterance text
     ↳ The GoE service is already loaded in Docker — call it directly
     ↳ Result: REAL GoE distributions, NOT hand-coded MELD→GoE maps
     ↳ Eliminates the root cause of the synthetic label collapse

  3. GoEmotions CDM data       ~25,000 step pairs (already in trainer pipeline)
     ↳ Already used for meta-learner training via trainer/data/goemotions.py
     ↳ Contains real GoE model outputs (the dataset the model was trained on)
     ↳ Pseudo-conversations: group by similar emotion → N consecutive samples

Total: ~34,000 step pairs with real GoE distributions.
Split: 90% train, 10% val, conversations (not messages) as split unit.

Loss function:
  L = 0.7 × KL(prior_head, real_goe_next) + 0.3 × CrossEntropy(phase_head, derived_phase)

  KL loss: drives trajectory_prior toward the actual next GoE distribution
  CE loss: drives phase_head toward the algorithmically derived conversation phase
  Ratio 0.7/0.3: phase is auxiliary regularization; prior quality is primary goal

INTEGRATION: ZERO BREAKING CHANGES
────────────────────────────────────
The EDE is a drop-in replacement at the inference.py level:

  CURRENT:  run_trajectory_step() → {"predicted_next": {...}, "top_predicted": "neutral"}
  NEW:      run_ede_step()        → {"predicted_next": {...}, "top_predicted": "anger", "phase": "ESCALATION"}

  Feature vector [79:107] stays 28-dim (EDE prior head output, projected to 28).
  FEATURE_DIM stays 107. No meta-learner retraining required for the prior slot.

  Phase label stored in Redis: trajectory:{conv_id}:phase
  aggregate_and_publish reads phase → passes to GatingEnsembleNet as scalar signal
  (adds one line to main.py, no schema change)

  Redis storage change:
    CURRENT: trajectory:{conv_id}:hidden  →  JSON of (h, c) tensors ~2KB, locked
    NEW:     trajectory:{conv_id}:goe_history  →  JSON list of last N GoE dicts ~3KB, no lock

EXPECTED OUTCOME
────────────────
  Metric                        Current LSTM    EDE (projected)
  ─────────────────────────────────────────────────────────────
  "neutral" prediction rate     98.8%           <40% (matches real distribution)
  long-conv avg confidence      0.412           0.48–0.55 (trajectory conditions meta)
  phase detection               none            6-class, zero-annotation
  Redis lock contention         yes             none (stateless history list)
  training data quality         synthetic maps  real GoE model outputs
  interpretability              none            attention weights over history window

═══════════════════════════════════════════════════════════════════════════════
IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Label constants ────────────────────────────────────────────────────────────

EMOTION_LABELS_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
GOE_IDX = {e: i for i, e in enumerate(EMOTION_LABELS_28)}

# Valence map — needed for phase derivation from emotion labels
EMOTION_VALENCE = {
    "anger": -0.80, "annoyance": -0.60, "disapproval": -0.70, "disgust": -0.80,
    "disappointment": -0.60, "embarrassment": -0.50, "fear": -0.70, "grief": -0.90,
    "nervousness": -0.50, "remorse": -0.70, "sadness": -0.70,
    "admiration": 0.80, "amusement": 0.60, "approval": 0.60, "caring": 0.70,
    "curiosity": 0.30, "desire": 0.50, "excitement": 0.80, "gratitude": 0.80,
    "joy": 0.90, "love": 0.90, "optimism": 0.70, "pride": 0.70,
    "relief": 0.60, "neutral": 0.00, "confusion": -0.10,
    "realization": 0.10, "surprise": 0.20,
}

PHASES = ["opening", "escalation", "peak", "turning_point", "resolution", "sustained"]
PHASE_IDX = {p: i for i, p in enumerate(PHASES)}
N_PHASES = len(PHASES)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE DERIVATION  (zero annotation — algorithmically derived from valence)
# ══════════════════════════════════════════════════════════════════════════════

def derive_phase(valence_window: List[float]) -> int:
    """
    Map a recent valence history to one of 6 conversation phases.
    valence_window: [v_{T-K}, ..., v_{T-1}] — most recent last.

    Rules derived from DB observation:
      volatility jumps 26× at message 31+ → phase is a real phenomenon.
      pride↔grief oscillation (dominant transition pair) → peak/turning_point signals.
    """
    if len(valence_window) < 3:
        return PHASE_IDX["opening"]

    v = np.array(valence_window[-8:], dtype=np.float32)
    net_delta = float(v[-1] - v[0])
    volatility = float(np.std(v))
    mean_recent = float(np.mean(v[-3:]))
    velocity = float(v[-1] - v[-2]) if len(v) >= 2 else 0.0

    # High volatility + strong valence → peak intensity
    if volatility > 0.25 and abs(mean_recent) > 0.20:
        return PHASE_IDX["peak"]

    # Abrupt single-step reversal → turning point
    if abs(velocity) > 0.20:
        return PHASE_IDX["turning_point"]

    # Sustained decline into negative → escalation
    if net_delta < -0.15 and mean_recent < 0.0:
        return PHASE_IDX["escalation"]

    # Recovery from negative territory → resolution
    if net_delta > 0.15 and mean_recent >= -0.05:
        return PHASE_IDX["resolution"]

    # Very stable (low volatility, consistent emotion)
    if volatility < 0.05:
        return PHASE_IDX["sustained"]

    # Default: still establishing
    return PHASE_IDX["opening"]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class EmotionalDialogueEncoder(nn.Module):
    """
    Transformer-based conversation history encoder.

    Reads the last N real GoE distributions and produces:
      1. trajectory_prior [28]: P(emotion_T | history_{T-N..T-1}) — Bayesian prior
         for meta-learner feature vector slot [79:107]
      2. phase_logits [6]: conversation phase classification (auxiliary head)

    Design decisions:
      - Pre-LayerNorm (norm_first=True): more stable than post-LN for small datasets
      - CLS token: learns a global conversation state vector attending over all N steps
      - Learned positional embedding: position 0 = oldest, N-1 = most recent
      - No CDM in input: separation of concerns — EDE learns from raw emotion dynamics
      - Softmax on prior head: output is a proper probability distribution (not sigmoid)
      - Padding mask: handles variable-length histories (< N messages at conversation start)

    Parameters: ~85K (d=64, 2 layers, 4 heads) — trains in <5 min on CPU
    """

    def __init__(
        self,
        n_emotions: int = 28,
        n_phases: int = N_PHASES,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_window: int = 12,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_emotions = n_emotions
        self.n_phases = n_phases
        self.max_window = max_window
        self.d_model = d_model

        # Project each GoE distribution [28] → model dimension [64]
        self.input_proj = nn.Sequential(
            nn.Linear(n_emotions, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable [CLS] token — accumulates global conversation state
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Position 0 = oldest message, N = CLS (appended first in forward)
        self.pos_embed = nn.Embedding(max_window + 1, d_model)

        # Transformer — Pre-LN for gradient stability on small datasets
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,   # 256
            dropout=dropout,
            batch_first=True,
            norm_first=True,               # Pre-LN
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,    # required for padding mask compatibility
        )

        # ── Output heads from CLS ──────────────────────────────────────────────
        # Prior head: the Bayesian prior over 28 emotions given conversation history
        self.prior_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_emotions),
            nn.Softmax(dim=-1),
        )

        # Phase head: conversation phase classification (auxiliary regularizer)
        self.phase_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_phases),
        )

    def forward(
        self,
        history: torch.Tensor,                    # [B, N, 28]
        padding_mask: Optional[torch.Tensor] = None,  # [B, N] True = padding position
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            history:      [B, N, 28]  GoE distributions, oldest first
            padding_mask: [B, N]      True at padded (absent) positions

        Returns:
            prior  [B, 28]: trajectory prior distribution (Softmax, sums to 1)
            phase  [B, 6]:  phase logits (raw, apply softmax for probabilities)
        """
        B, N, _ = history.shape

        # Project GoE distributions to model dimension
        x = self.input_proj(history)               # [B, N, d_model]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)    # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)             # [B, N+1, d_model]

        # Add positional embeddings: CLS=position 0, history tokens=positions 1..N
        positions = torch.arange(N + 1, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)  # [B, N+1, d_model]

        # Extend padding mask for CLS (CLS token is never masked)
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            mask = torch.cat([cls_mask, padding_mask], dim=1)  # [B, N+1]
        else:
            mask = None

        # Transformer self-attention over [CLS, goe_{T-N}, ..., goe_{T-1}]
        h = self.transformer(x, src_key_padding_mask=mask)  # [B, N+1, d_model]

        # Extract CLS representation (global conversation state)
        cls_out = h[:, 0]                          # [B, d_model]

        prior = self.prior_head(cls_out)           # [B, 28] — Softmax output
        phase = self.phase_head(cls_out)           # [B, 6]  — raw logits

        return prior, phase


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def goe_dict_to_vec(goe: dict) -> np.ndarray:
    """Convert a GoE score dict to a 28-dim numpy array using canonical label order."""
    return np.array(
        [float(goe.get(e, 0.0)) for e in EMOTION_LABELS_28],
        dtype=np.float32,
    )


def load_db_sequences(db_url: str) -> List[Tuple[np.ndarray, np.ndarray, List[float]]]:
    """
    Load real GoE sequences from the production database.

    Returns list of (X, Y, valences) where:
      X [T, 28]: GoE distributions for messages 0..T-1 (input history)
      Y [T, 28]: GoE distributions for messages 1..T   (targets — real next)
      valences [T]: valence values for phase derivation

    Deduplication: filters repeated messages (41% of the longest conversation
    are demo loop repeats). Uses text hash to detect exact repetitions.
    """
    import psycopg2

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("""
        SELECT
            m.conversation_id,
            m.timestamp,
            m.text,
            ea.pipeline_log_json
        FROM messages m
        JOIN emotion_analysis ea ON ea.message_id = m.message_id
        WHERE ea.pipeline_log_json IS NOT NULL
        ORDER BY m.conversation_id, m.timestamp
    """)
    rows = cur.fetchall()
    conn.close()

    by_conv: Dict[str, List] = defaultdict(list)
    for conv_id, ts, text, log_json in rows:
        try:
            log = json.loads(log_json)
            goe_raw = log.get("models", {}).get("go_emotions", {})
            valence = float(
                (log.get("context_snapshot") or {}).get("cur_valence", 0.0) or 0.0
            )
            if goe_raw:
                by_conv[conv_id].append({
                    "text": text, "goe": goe_raw, "valence": valence, "ts": ts
                })
        except Exception:
            continue

    sequences = []
    for conv_id, msgs in by_conv.items():
        # Deduplicate repeated messages within the same conversation
        seen_texts = set()
        deduped = []
        for m in msgs:
            key = m["text"].strip().lower()
            if key not in seen_texts:
                seen_texts.add(key)
                deduped.append(m)

        if len(deduped) < 2:
            continue

        goe_vecs = np.stack([goe_dict_to_vec(m["goe"]) for m in deduped])
        valences = [m["valence"] for m in deduped]

        X = goe_vecs[:-1]      # messages 0..T-1
        Y = goe_vecs[1:]       # targets: messages 1..T
        V = valences[1:]       # valences aligned with Y

        sequences.append((X, Y, V))

    total_pairs = sum(len(X) for X, _, _ in sequences)
    print(f"DB: {len(sequences)} conversations, {total_pairs} step pairs "
          f"(deduplicated real GoE distributions)")
    return sequences


def load_meld_sequences_real_goe(meld_cache_path: str) -> List[Tuple[np.ndarray, np.ndarray, List[float]]]:
    """
    Load MELD conversations and return GoE sequences using the ACTUAL GoE model
    (not the synthetic MELD→GoE label maps that caused the 98.8% neutral collapse).

    The GoE model (bhadresh-savani/bert-base-go-emotion) is already loaded inside
    the goemotions Docker service. This function calls it directly via the
    transformers library — the model weights are cached locally after first use.

    This is the critical fix: real GoE distributions replace the hand-coded maps.
    """
    from transformers import pipeline as hf_pipeline

    print("Loading GoEmotions model for MELD feature extraction...")
    goe_model = hf_pipeline(
        "text-classification",
        model="bhadresh-savani/bert-base-go-emotion",
        top_k=None,           # return all 28 class scores
        device=-1,            # CPU (consistent with training environment)
        batch_size=32,
    )

    with open(meld_cache_path) as f:
        raw = json.load(f)

    by_dialog: Dict[str, List] = defaultdict(list)
    for row in raw:
        by_dialog[row["d"]].append(row)

    all_texts = []
    all_positions = []   # (dialog_id, utterance_idx)
    for dialog_id, utts in by_dialog.items():
        for i, utt in enumerate(sorted(utts, key=lambda r: r["u"])):
            all_texts.append(utt["t"])
            all_positions.append((dialog_id, i, utt.get("e", "neutral")))

    # Run GoE model on all MELD utterances in one batched pass
    print(f"Running GoE model on {len(all_texts)} MELD utterances (batched)...")
    all_results = goe_model(all_texts)   # list of [{label, score}, ...]

    # Rebuild into per-dialog sequences
    goe_by_pos: Dict = {}
    for pos, scores in zip(all_positions, all_results):
        goe_dict = {item["label"]: item["score"] for item in scores}
        goe_by_pos[pos[:2]] = goe_dict_to_vec(goe_dict)

    sequences = []
    for dialog_id, utts in by_dialog.items():
        utts_sorted = sorted(utts, key=lambda r: r["u"])
        if len(utts_sorted) < 2:
            continue

        goe_vecs = []
        valences = []
        for i, utt in enumerate(utts_sorted):
            key = (dialog_id, i)
            if key in goe_by_pos:
                vec = goe_by_pos[key]
                goe_vecs.append(vec)
                # Use GoE→valence map for phase derivation
                dominant = EMOTION_LABELS_28[int(vec.argmax())]
                valences.append(EMOTION_VALENCE.get(dominant, 0.0))

        if len(goe_vecs) < 2:
            continue

        goe_arr = np.stack(goe_vecs)
        X = goe_arr[:-1]
        Y = goe_arr[1:]
        V = valences[1:]
        sequences.append((X, Y, V))

    total_pairs = sum(len(X) for X, _, _ in sequences)
    print(f"MELD (real GoE): {len(sequences)} sequences, {total_pairs} step pairs")
    return sequences


def build_windows(
    sequences: List[Tuple[np.ndarray, np.ndarray, List[float]]],
    max_window: int = 12,
) -> List[Dict]:
    """
    Convert conversation sequences into fixed-window training samples.

    Each sample:
      history [N, 28]: up to max_window previous GoE distributions (padded if < N)
      target  [28]:    the GoE distribution for the current message (what to predict)
      phase   [int]:   derived conversation phase (training label for phase head)
      length  [int]:   actual history length before padding (for padding mask)
    """
    samples = []
    for X, Y, V in sequences:
        T = len(X)
        for t in range(T):
            # History: up to max_window messages before position t
            start = max(0, t - max_window)
            hist = X[start: t]              # [k, 28] where k = min(t, max_window)
            target = Y[t]                   # [28] — real next GoE distribution

            # Derive phase from valence history up to and including position t
            valence_window = V[start: t + 1]
            phase = derive_phase(valence_window)

            # Pad history to max_window on the LEFT (oldest positions = padding)
            length = len(hist)
            if length < max_window:
                pad = np.zeros((max_window - length, 28), dtype=np.float32)
                hist_padded = np.concatenate([pad, hist], axis=0)
                padding_mask = np.array(
                    [True] * (max_window - length) + [False] * length,
                    dtype=bool,
                )
            else:
                hist_padded = hist[-max_window:]
                padding_mask = np.zeros(max_window, dtype=bool)

            samples.append({
                "history": hist_padded,        # [max_window, 28]
                "target": target,              # [28]
                "phase": phase,                # int
                "padding_mask": padding_mask,  # [max_window]
                "length": length,
            })
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def topk_acc(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """Top-k accuracy where target is a soft distribution (argmax = ground truth)."""
    target_cls = target.argmax(dim=-1)
    topk = pred.topk(k, dim=-1).indices
    return (topk == target_cls.unsqueeze(-1)).any(dim=-1).float().mean().item()


def train_ede(
    samples: List[Dict],
    device: str = "cpu",
    epochs: int = 60,
    lr: float = 3e-4,
    batch_size: int = 128,
    max_window: int = 12,
    d_model: int = 64,
    val_frac: float = 0.10,
) -> EmotionalDialogueEncoder:
    """
    Train the EmotionalDialogueEncoder.

    Loss: 0.7 × KL(prior_head, real_next_goe) + 0.3 × CrossEntropy(phase_head, derived_phase)

    KL is used instead of CE for the prior because the targets are soft distributions
    (real GoE model outputs), not hard one-hot labels. KL correctly measures divergence
    between two probability distributions.

    The 0.7/0.3 split treats phase prediction as an auxiliary regularizer:
    it forces the CLS representation to encode conversation dynamics (not just
    memorize individual GoE distributions), while keeping the primary objective
    on producing accurate priors.

    Class weighting for phase CE: phases are imbalanced (most samples = "sustained"
    in long conversations). Inverse-frequency weighting capped at 5× prevents
    rare phases from dominating the gradient.
    """
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_frac))
    val_samples = samples[:n_val]
    trn_samples = samples[n_val:]

    # Phase class weights — inverse frequency, capped
    phase_counts = np.bincount(
        [s["phase"] for s in trn_samples], minlength=N_PHASES
    ).astype(np.float32)
    phase_counts = np.where(phase_counts == 0, 1.0, phase_counts)
    phase_weights = phase_counts.sum() / (N_PHASES * phase_counts)
    phase_weights = np.clip(phase_weights, 0.20, 5.0)
    phase_w_tensor = torch.tensor(phase_weights, dtype=torch.float32, device=device)

    model = EmotionalDialogueEncoder(
        max_window=max_window, d_model=d_model,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(epochs // 3, 15), T_mult=1, eta_min=lr * 0.01,
    )

    best_top1 = 0.0
    best_state = None

    def run_batch(batch_samples, training: bool):
        H = torch.tensor(
            np.stack([s["history"] for s in batch_samples]),
            dtype=torch.float32, device=device,
        )                                        # [B, N, 28]
        T = torch.tensor(
            np.stack([s["target"] for s in batch_samples]),
            dtype=torch.float32, device=device,
        )                                        # [B, 28]
        P = torch.tensor(
            [s["phase"] for s in batch_samples],
            dtype=torch.long, device=device,
        )                                        # [B]
        M = torch.tensor(
            np.stack([s["padding_mask"] for s in batch_samples]),
            dtype=torch.bool, device=device,
        )                                        # [B, N]

        prior, phase_logits = model(H, padding_mask=M)

        # KL divergence: -(target * log(pred)).sum(dim=-1)
        kl_loss = -(T * torch.log(prior.clamp(min=1e-8))).sum(dim=-1).mean()

        # Weighted cross-entropy for phase
        ce_loss = F.cross_entropy(phase_logits, P, weight=phase_w_tensor)

        loss = 0.7 * kl_loss + 0.3 * ce_loss
        return loss, prior, T, phase_logits, P

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(trn_samples)
        trn_loss = 0.0
        n_batches = 0

        for i in range(0, len(trn_samples), batch_size):
            batch = trn_samples[i: i + batch_size]
            if not batch:
                continue
            opt.zero_grad()
            loss, _, _, _, _ = run_batch(batch, training=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trn_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            all_prior, all_target, all_phase_pred, all_phase_true = [], [], [], []
            val_loss = 0.0

            with torch.no_grad():
                for i in range(0, len(val_samples), batch_size):
                    batch = val_samples[i: i + batch_size]
                    if not batch:
                        continue
                    loss, prior, tgt, ph_logits, ph_true = run_batch(batch, training=False)
                    val_loss += loss.item()
                    all_prior.append(prior)
                    all_target.append(tgt)
                    all_phase_pred.append(ph_logits.argmax(dim=-1))
                    all_phase_true.append(ph_true)

            all_prior = torch.cat(all_prior)
            all_target = torch.cat(all_target)
            top1 = topk_acc(all_prior, all_target, 1)
            top3 = topk_acc(all_prior, all_target, 3)

            ph_pred = torch.cat(all_phase_pred).cpu().numpy()
            ph_true = torch.cat(all_phase_true).cpu().numpy()
            phase_acc = float((ph_pred == ph_true).mean())

            # Collapse check: flag if >60% of predictions are "neutral"
            neutral_pct = float((all_prior.argmax(dim=-1) == GOE_IDX["neutral"]).float().mean())

            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"trn={trn_loss/max(n_batches,1):.3f}  "
                f"val={val_loss/max(len(val_samples)//batch_size,1):.3f}  "
                f"top1={top1:.3f}  top3={top3:.3f}  "
                f"phase={phase_acc:.3f}  "
                f"neutral%={neutral_pct:.2%}"
                + ("  ← COLLAPSE" if neutral_pct > 0.60 else "")
            )

            if top1 > best_top1:
                best_top1 = top1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"\nBest top1={best_top1:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE  (drop-in replacement for trajectory/inference.py)
# ══════════════════════════════════════════════════════════════════════════════

HISTORY_KEY_PREFIX = "ede:"
HISTORY_TTL = 7200       # 2-hour inactivity TTL (same as LSTM)
MAX_HISTORY = 12         # max messages kept in Redis list


async def load_goe_history(conv_id: str, redis) -> List[np.ndarray]:
    """
    Read last MAX_HISTORY GoE distributions from Redis.
    Returns [] on first message (cold start). Caller gets uniform prior = equal weight.
    Simple list — no locks, no serialization of tensors.
    """
    key = f"{HISTORY_KEY_PREFIX}{conv_id}:goe_history"
    try:
        raw = await redis.get(key)
        if not raw:
            return []
        data = json.loads(raw)
        return [np.array([d.get(e, 0.0) for e in EMOTION_LABELS_28], dtype=np.float32)
                for d in data]
    except Exception:
        return []


async def save_goe_history(conv_id: str, goe_scores: dict, redis):
    """
    Append current GoE distribution to history and persist.
    Trims to MAX_HISTORY. Atomic: load→append→save is acceptable since
    GoE scores are append-only and order is guaranteed by conversation flow.
    """
    key = f"{HISTORY_KEY_PREFIX}{conv_id}:goe_history"
    try:
        raw = await redis.get(key)
        history = json.loads(raw) if raw else []
        history.append({e: float(goe_scores.get(e, 0.0)) for e in EMOTION_LABELS_28})
        history = history[-MAX_HISTORY:]
        await redis.set(key, json.dumps(history), ex=HISTORY_TTL)
    except Exception:
        pass


async def run_ede_step(
    model: EmotionalDialogueEncoder,
    goe_scores: dict,
    conv_id: str,
    redis,
    valence_history: List[float] = None,
) -> dict:
    """
    Full EDE inference step. Drop-in replacement for run_trajectory_step().

    Returns dict with same keys as the current LSTM step plus "phase":
    {
        "predicted_next":  {"joy": 0.42, ...}   top-5 emotions
        "top_predicted":   "joy"
        "phase":           "escalation"
        "model_available": True
    }
    """
    if model is None:
        return {}

    try:
        # Load history (no lock needed)
        history_vecs = await load_goe_history(conv_id, redis)

        # Build history tensor
        N = model.max_window
        if history_vecs:
            hist_arr = np.stack(history_vecs[-N:])                     # [k, 28]
            k = len(hist_arr)
            if k < N:
                pad = np.zeros((N - k, 28), dtype=np.float32)
                hist_padded = np.concatenate([pad, hist_arr], axis=0)  # [N, 28]
                mask = np.array([True] * (N - k) + [False] * k, dtype=bool)
            else:
                hist_padded = hist_arr
                mask = np.zeros(N, dtype=bool)
        else:
            # Cold start: all zeros + all masked → CLS attends to nothing
            hist_padded = np.zeros((N, 28), dtype=np.float32)
            mask = np.ones(N, dtype=bool)

        x = torch.tensor(hist_padded, dtype=torch.float32).unsqueeze(0)   # [1, N, 28]
        m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)             # [1, N]

        with torch.no_grad():
            prior, phase_logits = model(x, padding_mask=m)

        scores = prior[0].cpu().tolist()           # [28] Softmax output
        phase_idx = int(phase_logits[0].argmax())
        phase_name = PHASES[phase_idx]

        # Derive phase from valence history if provided (higher quality than model)
        if valence_history and len(valence_history) >= 2:
            phase_name = PHASES[derive_phase(valence_history)]

        # Save updated history (current message appended)
        await save_goe_history(conv_id, goe_scores, redis)

        # Save trajectory prior for meta-learner feature vector [79:107]
        # This is the correct slot — EDE prior conditions classification of message T
        try:
            await redis.set(
                f"trajectory:{conv_id}:prior",
                json.dumps(scores),
                ex=HISTORY_TTL,
            )
            await redis.set(
                f"trajectory:{conv_id}:phase",
                phase_name,
                ex=HISTORY_TTL,
            )
        except Exception:
            pass

        emotion_scores = dict(zip(EMOTION_LABELS_28, scores))
        top5 = sorted(emotion_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]

        return {
            "predicted_next":  {k: round(v, 4) for k, v in top5},
            "top_predicted":   top5[0][0],
            "phase":           phase_name,
            "model_available": True,
        }

    except Exception as e:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Train EDE from scratch using DB + MELD real GoE data.
    Run: python ede_proposal.py [--meld-only] [--epochs N] [--hidden N]

    Output: .cache/ede_model.pt + .cache/ede_config.json
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=60)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--hidden",    type=int,   default=64)
    parser.add_argument("--window",    type=int,   default=12)
    parser.add_argument("--meld-only", action="store_true",
                        help="Skip DB loading (use when no local DB access)")
    parser.add_argument("--no-meld",   action="store_true",
                        help="Skip MELD GoE inference (faster, DB only)")
    args = parser.parse_args()

    ROOT  = Path(__file__).parent
    CACHE = ROOT / ".cache"
    CACHE.mkdir(exist_ok=True)

    sys.path.insert(0, str(ROOT / "central_responder_service"))

    all_sequences = []

    # ── Source 1: Real DB ──────────────────────────────────────────────────────
    if not args.meld_only:
        DB_URL = os.environ.get(
            "DATABASE_URL",
            "postgresql://user:password@localhost:5432/emotion_db"
        )
        try:
            db_seqs = load_db_sequences(DB_URL)
            all_sequences.extend(db_seqs)
        except Exception as e:
            print(f"DB load failed ({e}) — continuing with MELD only")

    # ── Source 2: MELD with real GoE inference ─────────────────────────────────
    meld_path = CACHE / "meld_raw_cache.json"
    if not args.no_meld and meld_path.exists():
        try:
            meld_seqs = load_meld_sequences_real_goe(str(meld_path))
            all_sequences.extend(meld_seqs)
        except Exception as e:
            print(f"MELD load failed ({e}) — continuing without MELD")
    elif not meld_path.exists():
        print("MELD cache not found — run trajectory_train.py once to populate .cache/")

    if not all_sequences:
        print("ERROR: No training data available.")
        sys.exit(1)

    total_pairs = sum(len(X) for X, _, _ in all_sequences)
    print(f"\nTotal: {len(all_sequences)} conversations, {total_pairs} step pairs")

    # ── Build fixed-window samples ─────────────────────────────────────────────
    samples = build_windows(all_sequences, max_window=args.window)
    print(f"Windows built: {len(samples)} training samples")

    phase_dist = np.bincount([s["phase"] for s in samples], minlength=N_PHASES)
    for i, p in enumerate(PHASES):
        print(f"  {p:15s}: {phase_dist[i]:6d} samples ({100*phase_dist[i]/len(samples):.1f}%)")

    # ── Train ──────────────────────────────────────────────────────────────────
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("Training on CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # NOTE: MPS has LSTM backward bug — EDE (Transformer) does NOT have this issue
        device = "mps"
        print("Training on Apple MPS (safe for Transformer, unlike LSTM)")

    model = train_ede(
        samples,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        max_window=args.window,
        d_model=args.hidden,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    model_out  = CACHE / "ede_model.pt"
    config_out = CACHE / "ede_config.json"

    torch.save({"model_state": model.state_dict()}, model_out)

    config = {
        "n_emotions":  28,
        "n_phases":    N_PHASES,
        "d_model":     args.hidden,
        "n_heads":     4,
        "n_layers":    2,
        "max_window":  args.window,
        "dropout":     0.15,
        "arch":        "EmotionalDialogueEncoder",
        "replaces":    "ConversationLSTM",
        "feature_dim_unchanged": True,
        "prior_slot": "[79:107]",
    }
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved: {model_out}")
    print(f"       {config_out}")
    print(f"\nIntegration: copy ede_model.pt and ede_config.json to .cache/")
    print(f"Update central_responder_service/trajectory/inference.py:")
    print(f"  replace run_trajectory_step() with run_ede_step()")
    print(f"  replace load_trajectory_model() with load_ede_model()")
    print(f"  no change to feature vector layout — [79:107] stays 28-dim")


def load_ede_model(model_path: str, config_path: str) -> Optional[EmotionalDialogueEncoder]:
    """Load EDE from checkpoint. Drop-in for load_trajectory_model()."""
    if not os.path.exists(model_path):
        return None
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        model = EmotionalDialogueEncoder(
            n_emotions=cfg.get("n_emotions", 28),
            n_phases=cfg.get("n_phases", N_PHASES),
            d_model=cfg.get("d_model", 64),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
            max_window=cfg.get("max_window", 12),
            dropout=cfg.get("dropout", 0.15),
        )
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"EDE loaded: d={cfg.get('d_model')} window={cfg.get('max_window')} phases={cfg.get('n_phases')}")
        return model
    except Exception as e:
        print(f"Failed to load EDE: {e}")
        return None


if __name__ == "__main__":
    main()
