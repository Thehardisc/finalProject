"""
aggregation_service/state/conversation_state.py — Conversation state aggregation logic.

Maintains per-conversation mood, valence, and dominant emotion in Redis.
Applies dynamic rule overrides and logs mood transitions.

Redis keys written per conversation:
  conversation:{id}          — existing state hash (valence, mood, counts)
  conversation:{id}:emotional_state — bot-readable summary (mood, trajectory, arc)
  conv:{id}:valence_hist     — existing sliding valence window (last 3)
  conv:{id}:mood_arc         — sliding mood window (last 5) — NEW
"""
import json
import os
import time
from datetime import datetime

from shared.utils.logger import get_logger
from aggregation_service.rules.dynamic_rules import handle_dynamic_rules

logger = get_logger("aggregation_service")

IDLE_TIMEOUT_SECONDS = int(os.environ.get("CONVERSATION_IDLE_MINUTES", "30")) * 60
IDLE_STREAM = "conversation_idle_stream"

# Dominant emotion → 8-class narrative mood
_EMOTION_TO_MOOD: dict = {
    'admiration': 'warm',      'amusement':      'joyful',
    'approval':   'warm',      'caring':         'warm',
    'curiosity':  'neutral',   'desire':         'warm',
    'excitement': 'joyful',    'gratitude':      'warm',
    'joy':        'joyful',    'love':           'warm',
    'optimism':   'joyful',    'pride':          'joyful',
    'relief':     'resolved',  'realization':    'neutral',
    'surprise':   'neutral',   'neutral':        'neutral',
    'anger':      'hostile',   'annoyance':      'hostile',
    'disapproval':'conflicted','disgust':        'hostile',
    'disappointment': 'melancholic',
    'embarrassment':  'conflicted',
    'fear':       'anxious',   'grief':          'melancholic',
    'nervousness':'anxious',   'remorse':        'melancholic',
    'sadness':    'melancholic','confusion':     'conflicted',
}


def _emotion_to_mood(dominant_emotion: str, valence: float) -> str:
    if dominant_emotion in _EMOTION_TO_MOOD:
        return _EMOTION_TO_MOOD[dominant_emotion]
    if valence >= 0.3:   return 'joyful'
    if valence >= 0.0:   return 'neutral'
    if valence >= -0.3:  return 'conflicted'
    if valence >= -0.6:  return 'melancholic'
    return 'hostile'


def _compute_trajectory(valence_history: list[float]) -> str:
    """Newest value first. Returns escalating / de-escalating / stable."""
    if len(valence_history) < 2:
        return 'stable'
    delta = valence_history[0] - valence_history[-1]
    if delta < -0.15:  return 'escalating'
    if delta >  0.15:  return 'de-escalating'
    return 'stable'


def _valence_to_mood(avg_valence: float) -> str:
    if avg_valence >= 0.6:    return "Ecstatic"
    if avg_valence >= 0.2:    return "Positive"
    if avg_valence >= 0.05:   return "Slightly Positive"
    if avg_valence >= -0.05:  return "Neutral"
    if avg_valence >= -0.2:   return "Slightly Negative"
    if avg_valence >= -0.6:   return "Negative"
    return "Hostile"


async def update_conversation_state(
    conversation_id: str, new_emotions: dict, r,
    original_text: str = "", user_id: str = ""
) -> dict:
    """
    Update the Redis conversation state hash for a conversation.
    Returns the new state dict.
    """
    state_key     = f"conversation:{conversation_id}"
    current_state = await r.hgetall(state_key)

    # ── Idle detection ────────────────────────────────────────────────────────
    # If the gap since the last message exceeds IDLE_TIMEOUT, the previous
    # session is considered complete. Publish to conversation_idle_stream so
    # api_service can run post-conversation analysis asynchronously.
    msg_count = int(current_state.get("message_count", 0))
    if msg_count > 0:
        last_activity = float(current_state.get("last_activity", 0.0))
        gap = time.time() - last_activity
        if gap > IDLE_TIMEOUT_SECONDS:
            try:
                await r.xadd(IDLE_STREAM, {
                    "conversation_id": conversation_id,
                    "idle_seconds":    str(round(gap)),
                    "message_count":   str(msg_count),
                })
                logger.info(
                    "conversation_idle",
                    extra={
                        "event":           "conversation_idle",
                        "conversation_id": conversation_id,
                        "idle_seconds":    round(gap),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to publish idle event: {e}")

    msg_count   = int(current_state.get("message_count", 0))
    ema_valence = float(current_state.get("ema_valence", 0.0))  # replaces accumulated_valence
    prev_mood   = current_state.get("overall_mood", "Neutral")

    # Dynamic rule check
    override_trigger, override_meaning = await handle_dynamic_rules(
        conversation_id, original_text, r
    )

    new_valence      = new_emotions.get("vader_compound", 0.0)
    dominant_emotion = new_emotions.get("dominant_emotion")
    dominant_score   = 1.0

    if not dominant_emotion:
        latest = {k: v for k, v in new_emotions.items()
                  if k not in ["vader_neg", "vader_neu", "vader_pos",
                               "vader_compound", "dominant_emotion"]}
        dominant_emotion = "None"
        dominant_score   = 0.0
        for emo, score in latest.items():
            if score > dominant_score:
                dominant_score   = score
                dominant_emotion = emo

    # Apply override
    if override_meaning:
        logger.info(f"Applying override logic for meaning: {override_meaning}")
        m = override_meaning.lower()
        if any(w in m for w in ["love", "happy", "good", "great", "joy"]):
            new_valence      = 0.9
            dominant_emotion = "love"
            dominant_score   = 1.0
        elif any(w in m for w in ["hate", "bad", "angry", "kill", "sad"]):
            new_valence      = -0.9
            dominant_emotion = "anger"
            dominant_score   = 1.0

    msg_count += 1
    # Exponential moving average — recent messages have higher weight.
    # Alpha=0.35: each new message contributes 35%, history contributes 65%.
    # After 5 consecutive angry messages the EMA reflects ~83% of that anger.
    alpha       = 0.35
    avg_valence = alpha * new_valence + (1.0 - alpha) * ema_valence
    overall_mood = _valence_to_mood(avg_valence)

    if overall_mood != prev_mood:
        logger.info(f"[State Shift] {conversation_id}: MOOD TRANSITION | "
                    f"{prev_mood} -> {overall_mood}")
        logger.info(f"             Reasoning: EMA Valence = {avg_valence:.4f}")

    now = time.time()
    new_state = {
        "message_count":          msg_count,
        "ema_valence":            avg_valence,   # EMA — used as next message's context
        "average_valence":        avg_valence,   # kept for API compatibility
        "overall_mood":           overall_mood,
        "dominant_emotion":       dominant_emotion,
        "dominant_emotion_score": dominant_score,
        "last_updated":           now,
        "last_activity":          now,           # used by idle detection
        "last_message_emotions":  json.dumps(new_emotions)
    }

    # CDM history — written here so context_engine reads it for the NEXT message
    # prev_valence lets context_engine compute velocity without a separate query
    new_state["prev_valence"] = str(ema_valence)   # valence BEFORE this update

    # Per-speaker valence for speaker-asymmetry feature
    if user_id:
        new_state[f"spk:{user_id}"] = str(new_valence)

    await r.hset(state_key, mapping={k: str(v) for k, v in new_state.items()})
    await r.expire(state_key, 86400 * 7)  # 7-day TTL — stale conversations don't leak memory

    # Sliding valence history (last 3) — used by context_engine for velocity + acceleration
    hist_key = f"conv:{conversation_id}:valence_hist"
    await r.lpush(hist_key, str(new_valence))
    await r.ltrim(hist_key, 0, 2)
    await r.expire(hist_key, 86400 * 7)

    # ── Emotional state for bot consumption ──────────────────────────────────
    # mood_arc: sliding window of last 5 narrative moods (newest first)
    arc_key      = f"conv:{conversation_id}:mood_arc"
    current_mood = _emotion_to_mood(dominant_emotion, new_valence)
    await r.lpush(arc_key, current_mood)
    await r.ltrim(arc_key, 0, 4)
    await r.expire(arc_key, 86400 * 7)
    mood_arc = await r.lrange(arc_key, 0, -1)  # newest first

    # trajectory from valence history (read back what we just stored)
    raw_hist = await r.lrange(hist_key, 0, -1)
    valence_history = [float(v) for v in raw_hist]
    trajectory = _compute_trajectory(valence_history)

    emotional_state = {
        "current_mood":      current_mood,
        "dominant_emotion":  dominant_emotion,
        "trajectory":        trajectory,
        "arc":               json.dumps(mood_arc),
        "ema_valence":       round(avg_valence, 4),
        "message_count":     msg_count,
        "last_updated":      time.time(),
    }
    es_key = f"conversation:{conversation_id}:emotional_state"
    await r.hset(es_key, mapping={k: str(v) for k, v in emotional_state.items()})
    await r.expire(es_key, 86400 * 7)

    logger.log_stats(f"AGGREGATION: {conversation_id}", {
        "Session ID":    conversation_id,
        "Total Messages": msg_count,
        "Last Valence":  f"{new_valence:+.4f}",
        "Avg Valence":   f"{avg_valence:.4f}",
        "Mood":          overall_mood,
        "Dominant":      f"{dominant_emotion} ({dominant_score:.2%})",
        "Heartbeat":     datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    })

    return new_state
