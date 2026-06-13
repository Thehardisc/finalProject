"""
Trajectory inference — EmotionalDialogueEncoder (EDE) integration.

Public API (unchanged from LSTM version):
  run_trajectory_step(model, model_outputs, conv_id, redis, context_vector=None)
  load_trajectory_model(model_path, config_path)

Changes from LSTM:
  - No hidden state serialization (no h/c tensors, no distributed lock)
  - Redis stores plain JSON list of last N GoE distributions (trajectory:{conv_id}:goe_history)
  - Trajectory prior [79:107] is still written to trajectory:{conv_id}:prior
  - Phase label written to trajectory:{conv_id}:phase
"""

import json
import numpy as np
from typing import Optional, List

import torch

from shared.utils.logger import get_logger

logger = get_logger("trajectory")

EMOTION_LABELS_28 = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral',
]

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

HISTORY_KEY_PREFIX = "trajectory:"
HISTORY_TTL        = 7200   # 2-hour inactivity TTL
MAX_HISTORY        = 12


def _goe_dict_to_vec(goe: dict) -> np.ndarray:
    return np.array([float(goe.get(e, 0.0)) for e in EMOTION_LABELS_28], dtype=np.float32)


def _goe_vec_to_valence(vec: np.ndarray) -> float:
    return float(sum(vec[i] * _EMOTION_VALENCE.get(e, 0.0)
                     for i, e in enumerate(EMOTION_LABELS_28)))


def _derive_phase(valence_window: List[float]) -> str:
    from trajectory.model import PHASES
    if len(valence_window) < 3:
        return "opening"

    v           = np.array(valence_window[-8:], dtype=np.float32)
    net_delta   = float(v[-1] - v[0])
    volatility  = float(np.std(v))
    mean_recent = float(np.mean(v[-3:]))
    velocity    = float(v[-1] - v[-2]) if len(v) >= 2 else 0.0

    if volatility > 0.25 and abs(mean_recent) > 0.20:
        return "peak"
    if abs(velocity) > 0.20:
        return "turning_point"
    if net_delta < -0.15 and mean_recent < 0.0:
        return "escalation"
    if net_delta > 0.15 and mean_recent >= -0.05:
        return "resolution"
    if volatility < 0.05:
        return "sustained"
    return "opening"


async def _load_goe_history(conv_id: str, redis) -> List[np.ndarray]:
    key = f"{HISTORY_KEY_PREFIX}{conv_id}:goe_history"
    try:
        raw = await redis.get(key)
        if not raw:
            return []
        data = json.loads(raw)
        return [_goe_dict_to_vec(d) for d in data]
    except Exception:
        return []


async def _save_goe_history(conv_id: str, goe_scores: dict, redis):
    key = f"{HISTORY_KEY_PREFIX}{conv_id}:goe_history"
    try:
        raw = await redis.get(key)
        history = json.loads(raw) if raw else []
        history.append({e: float(goe_scores.get(e, 0.0)) for e in EMOTION_LABELS_28})
        history = history[-MAX_HISTORY:]
        await redis.set(key, json.dumps(history), ex=HISTORY_TTL)
    except Exception as e:
        logger.warning(f"Failed to save GoE history for {conv_id}: {e}")


async def run_trajectory_step(
    model,
    model_outputs: dict,
    conv_id: str,
    redis,
    context_vector: list = None,
) -> dict:
    """
    EDE inference step.  Drop-in for the old LSTM run_trajectory_step.

    Extracts go_emotions scores from model_outputs, runs EDE over stored
    history window, writes trajectory prior [79:107] to Redis, returns dict.
    """
    if model is None:
        return {}

    try:
        goe_scores = model_outputs.get("go_emotions", {})
        if not goe_scores:
            return {}

        history_vecs = await _load_goe_history(conv_id, redis)

        N = model.max_window
        if history_vecs:
            arr = np.stack(history_vecs[-N:])           # [k, 28]
            k   = len(arr)
            if k < N:
                pad     = np.zeros((N - k, 28), dtype=np.float32)
                hist    = np.concatenate([pad, arr], axis=0)
                mask    = np.array([True] * (N - k) + [False] * k, dtype=bool)
            else:
                hist    = arr
                mask    = np.zeros(N, dtype=bool)
        else:
            hist = np.zeros((N, 28), dtype=np.float32)
            mask = np.ones(N, dtype=bool)

        x = torch.tensor(hist, dtype=torch.float32).unsqueeze(0)   # [1, N, 28]
        m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)      # [1, N]

        with torch.no_grad():
            prior, phase_logits = model(x, padding_mask=m)

        scores    = prior[0].cpu().tolist()                         # [28]
        phase_idx = int(phase_logits[0].argmax())

        # Derive phase from valence computed from real GoE history if available
        if history_vecs:
            valence_window = [_goe_vec_to_valence(v) for v in history_vecs]
            phase_name     = _derive_phase(valence_window)
        else:
            from trajectory.model import PHASES
            phase_name = PHASES[phase_idx]

        # Save updated history (current message)
        await _save_goe_history(conv_id, goe_scores, redis)

        # Write trajectory prior for meta-learner [79:107]
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
        except Exception as pe:
            logger.warning(f"Failed to persist trajectory prior for {conv_id}: {pe}")

        emotion_scores = dict(zip(EMOTION_LABELS_28, scores))
        top5 = sorted(emotion_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]

        return {
            "predicted_next":  {k: round(v, 4) for k, v in top5},
            "top_predicted":   top5[0][0] if top5 else "neutral",
            "phase":           phase_name,
            "model_available": True,
        }

    except Exception as e:
        logger.warning(f"EDE inference failed for {conv_id}: {e}")
        return {}


def load_trajectory_model(model_path: str, config_path: str):
    """
    Load EmotionalDialogueEncoder from checkpoint.
    Returns model in eval mode, or None if files are missing.
    """
    import os
    from trajectory.model import EmotionalDialogueEncoder

    if not os.path.exists(model_path):
        logger.warning(f"EDE model not found at {model_path} — trajectory context disabled.")
        return None

    try:
        config = {}
        if os.path.exists(config_path):
            with open(config_path) as fh:
                config = json.load(fh)

        model = EmotionalDialogueEncoder(
            n_emotions  = config.get("n_emotions",  28),
            n_phases    = config.get("n_phases",    6),
            d_model     = config.get("d_model",     64),
            n_heads     = config.get("n_heads",     4),
            n_layers    = config.get("n_layers",    2),
            max_window  = config.get("max_window",  12),
            dropout     = config.get("dropout",     0.15),
        )

        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        logger.info(
            f"EDE loaded: d={config.get('d_model', 64)} "
            f"window={config.get('max_window', 12)} "
            f"top1={config.get('top1_acc', 0):.3f}"
        )
        return model

    except Exception as e:
        logger.error(f"Failed to load EDE model: {e}")
        return None
