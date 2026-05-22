"""
aggregation_service/state/conversation_state.py — Conversation state aggregation logic.

Maintains per-conversation mood, valence, and dominant emotion in Redis.
Applies dynamic rule overrides and logs mood transitions.
"""
import json
import time
from datetime import datetime

from shared.utils.logger import get_logger
from aggregation_service.rules.dynamic_rules import handle_dynamic_rules

logger = get_logger("aggregation_service")


def _valence_to_mood(avg_valence: float) -> str:
    if avg_valence >= 0.6:    return "Ecstatic"
    if avg_valence >= 0.2:    return "Positive"
    if avg_valence >= 0.05:   return "Slightly Positive"
    if avg_valence >= -0.05:  return "Neutral"
    if avg_valence >= -0.2:   return "Slightly Negative"
    if avg_valence >= -0.6:   return "Negative"
    return "Hostile"


async def update_conversation_state(
    conversation_id: str, new_emotions: dict, r, original_text: str = ""
) -> dict:
    """
    Update the Redis conversation state hash for a conversation.
    Returns the new state dict.
    """
    state_key     = f"conversation:{conversation_id}"
    current_state = await r.hgetall(state_key)

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

    new_state = {
        "message_count":          msg_count,
        "ema_valence":            avg_valence,   # EMA — used as next message's context
        "average_valence":        avg_valence,   # kept for API compatibility
        "overall_mood":           overall_mood,
        "dominant_emotion":       dominant_emotion,
        "dominant_emotion_score": dominant_score,
        "last_updated":           time.time(),
        "last_message_emotions":  json.dumps(new_emotions)
    }

    await r.hset(state_key, mapping={k: str(v) for k, v in new_state.items()})
    await r.expire(state_key, 86400 * 7)  # 7-day TTL — stale conversations don't leak memory

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
