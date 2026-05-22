"""
Trajectory inference module — integrates ConversationLSTM into the live pipeline.

Responsibilities:
  1. build_feature_vector()   — convert central_responder model_outputs → 67-dim tensor
  2. load_hidden_state()      — fetch (h, c) for a conversation from Redis
  3. save_hidden_state()      — persist (h, c) back to Redis with 7-day TTL
  4. run_trajectory_step()    — full inference: load → forward → save → return dict

The returned dict is merged into pipeline_log["trajectory"] by central_responder.
Falls back silently (returns empty dict) if model is None or any error occurs.
"""

import json
import logging
import numpy as np
from typing import Optional, Dict, Tuple

import torch

logger = logging.getLogger("trajectory")

# ── Label constants (must match training schema) ──────────────────────────────

EMOTION_LABELS_28 = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral',
]

BERT_LABELS_7  = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
VADER_KEYS_4   = ['neg', 'neu', 'pos', 'compound']

HIDDEN_KEY_PREFIX = "trajectory:"
HIDDEN_TTL        = 86400 * 7   # 7 days


# ── Feature construction ──────────────────────────────────────────────────────

def build_feature_vector(model_outputs: dict) -> torch.Tensor:
    """
    Convert central_responder model_outputs dict → [1, 1, 67] float tensor.
    model_outputs keys: go_emotions, basic_bert, vader, emojinet
    """
    go    = model_outputs.get("go_emotions", {})
    bert  = model_outputs.get("basic_bert",  {})
    vader = model_outputs.get("vader",       {})
    emoji = model_outputs.get("emojinet",    {})

    go_vec    = [float(go.get(e,    0.0)) for e in EMOTION_LABELS_28]   # 28
    bert_vec  = [float(bert.get(e,  0.0)) for e in BERT_LABELS_7]       # 7
    vader_vec = [float(vader.get(k, 0.0)) for k in VADER_KEYS_4]        # 4
    emoji_vec = [float(emoji.get(e, 0.0)) for e in EMOTION_LABELS_28]   # 28

    vec = go_vec + bert_vec + vader_vec + emoji_vec   # 67
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 67]


# ── Redis hidden state I/O ────────────────────────────────────────────────────

async def load_hidden_state(
    conv_id: str,
    redis,
    hidden_dim: int,
    num_layers: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load (h, c) from Redis for a conversation.
    Returns None if no state exists (first message in conversation).
    """
    key = f"{HIDDEN_KEY_PREFIX}{conv_id}:hidden"
    try:
        raw = await redis.get(key)
        if not raw:
            return None
        data = json.loads(raw)
        h = torch.tensor(data["h"], dtype=torch.float32)   # [num_layers, 1, hidden_dim]
        c = torch.tensor(data["c"], dtype=torch.float32)
        return (h, c)
    except Exception as e:
        logger.warning(f"Failed to load hidden state for {conv_id}: {e}")
        return None


async def save_hidden_state(
    conv_id: str,
    h_n: torch.Tensor,
    c_n: torch.Tensor,
    redis,
):
    """Persist (h, c) to Redis with 7-day TTL."""
    key = f"{HIDDEN_KEY_PREFIX}{conv_id}:hidden"
    try:
        data = {
            "h": h_n.detach().cpu().tolist(),
            "c": c_n.detach().cpu().tolist(),
        }
        await redis.set(key, json.dumps(data), ex=HIDDEN_TTL)
    except Exception as e:
        logger.warning(f"Failed to save hidden state for {conv_id}: {e}")


# ── Main inference step ───────────────────────────────────────────────────────

async def run_trajectory_step(
    model,
    model_outputs: dict,
    conv_id: str,
    redis,
) -> dict:
    """
    Full trajectory inference step for one message.

    Returns dict ready to be stored as pipeline_log["trajectory"]:
    {
        "predicted_next":  {"joy": 0.7, "anger": 0.3, ...}  top-5 predicted emotions
        "trajectory_vec":  [64 floats]                       conversation state embedding
        "top_predicted":   "joy"                             most likely next emotion
        "model_available": true
    }

    Returns {} if model is None or inference fails.
    """
    if model is None:
        return {}

    try:
        # 1. Build feature vector for this message
        x = build_feature_vector(model_outputs)

        # 2. Load previous hidden state (None = first message → LSTM uses zeros)
        hidden = await load_hidden_state(
            conv_id, redis,
            hidden_dim=model.hidden_dim,
            num_layers=model.num_layers,
        )

        # 3. Run one LSTM step
        next_emotions, traj_vec, (h_n, c_n) = model.predict_next(x, hidden=hidden)

        # 4. Save updated hidden state
        await save_hidden_state(conv_id, h_n, c_n, redis)

        # 5. Format output — top-5 predicted emotions for next message
        scores = next_emotions.cpu().tolist()
        emotion_scores = dict(zip(EMOTION_LABELS_28, scores))
        top5 = sorted(emotion_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top5_dict = {k: round(float(v), 4) for k, v in top5}
        top_predicted = top5[0][0] if top5 else "neutral"

        return {
            "predicted_next":  top5_dict,
            "top_predicted":   top_predicted,
            "trajectory_vec":  [round(float(v), 4) for v in traj_vec.cpu().tolist()],
            "model_available": True,
        }

    except Exception as e:
        logger.warning(f"Trajectory inference failed for {conv_id}: {e}")
        return {}


# ── Model loader ──────────────────────────────────────────────────────────────

def load_trajectory_model(model_path: str, config_path: str):
    """
    Load ConversationLSTM from checkpoint.
    Returns model in eval mode, or None if files are missing.
    """
    import os
    from trajectory.model import ConversationLSTM

    if not os.path.exists(model_path):
        logger.warning(f"Trajectory model not found at {model_path} — running without trajectory context.")
        return None

    try:
        # Load config to get architecture
        config = {}
        if os.path.exists(config_path):
            with open(config_path) as fh:
                config = json.load(fh)

        hidden_dim = config.get("hidden_dim", 64)
        num_layers = config.get("num_layers", 1)
        dropout    = config.get("dropout",    0.1)

        model = ConversationLSTM(
            input_dim=67,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=28,
            dropout=dropout,
        )

        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        logger.info(
            f"Trajectory model loaded: hidden={hidden_dim} layers={num_layers} "
            f"(epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', 0):.4f})"
        )
        return model

    except Exception as e:
        logger.error(f"Failed to load trajectory model: {e}")
        return None
