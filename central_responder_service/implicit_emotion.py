"""
implicit_emotion.py — Stage 1: Confidence-gated implicit emotion routing.

Two trigger paths:
  A. dominant_emotion == "neutral" (Stage 0 path)
     Override if implicit.confidence > IMPLICIT_CONF_THRESHOLD (0.50)

  B. meta_confidence < IMPLICIT_LOW_CONF_THRESHOLD (0.42) — meta-learner unsure
     Override if implicit.confidence > meta_confidence + IMPLICIT_LOW_CONF_MARGIN (0.10)
     so the implicit result must clearly beat the meta-learner score.

The responder is llm_reasoning_service (implicit_detector.py v2).
"""

import json
import os
from typing import Optional

from shared.utils.logger import get_logger

logger = get_logger("implicit_emotion")

IMPLICIT_EMOTION_ENABLED     = os.environ.get("IMPLICIT_EMOTION_ENABLED", "true").lower() == "true"
IMPLICIT_TIMEOUT_MS          = int(os.environ.get("IMPLICIT_TIMEOUT_MS", "300"))
IMPLICIT_MIN_WORDS           = int(os.environ.get("IMPLICIT_MIN_WORDS", "4"))
# Neutral path: skip if GoE is hyper-confident about neutral (genuinely neutral text)
IMPLICIT_GOE_NEUTRAL_CAP     = float(os.environ.get("IMPLICIT_GOE_NEUTRAL_CAP", "0.90"))
# Low-confidence path: also trigger when meta-learner confidence < this
IMPLICIT_LOW_CONF_THRESHOLD  = float(os.environ.get("IMPLICIT_LOW_CONF_THRESHOLD", "0.42"))
# Low-conf path: implicit result must beat meta_confidence by this margin to override
IMPLICIT_LOW_CONF_MARGIN     = float(os.environ.get("IMPLICIT_LOW_CONF_MARGIN", "0.10"))
# Neutral path: minimum implicit confidence to accept the override
IMPLICIT_CONF_THRESHOLD      = float(os.environ.get("IMPLICIT_CONF_THRESHOLD", "0.50"))

REQUEST_STREAM = "implicit_emotion_requests"


def is_implicit_candidate(
    dominant_emotion: str,
    goe_scores: dict,
    text: str,
    meta_confidence: float = 1.0,
) -> bool:
    """
    Returns True when the message warrants sending to the implicit detector.

    Path A (neutral gate): meta-learner said neutral + GoE not hyper-confident.
    Path B (low-conf gate): meta-learner said something but with very low confidence.
    """
    if not IMPLICIT_EMOTION_ENABLED:
        return False
    if len(text.split()) < IMPLICIT_MIN_WORDS:
        return False

    # Path A — neutral
    if dominant_emotion == "neutral":
        goe_neutral_conf = float(goe_scores.get("neutral", 0.0))
        return goe_neutral_conf <= IMPLICIT_GOE_NEUTRAL_CAP

    # Path B — low-confidence non-neutral prediction
    if meta_confidence < IMPLICIT_LOW_CONF_THRESHOLD:
        return True

    return False


def should_override(
    dominant_emotion: str,
    meta_confidence: float,
    impl_result: dict,
) -> bool:
    """
    Decide whether the implicit result should replace the meta-learner output.

    Neutral path: implicit must beat IMPLICIT_CONF_THRESHOLD.
    Low-conf path: implicit must beat meta_confidence by IMPLICIT_LOW_CONF_MARGIN.
    """
    impl_conf = float(impl_result.get("confidence", 0))
    impl_emo  = impl_result.get("emotion", "neutral")

    if impl_emo == "neutral":
        return False  # never override toward neutral

    if dominant_emotion == "neutral":
        return impl_conf > IMPLICIT_CONF_THRESHOLD

    # low-conf path
    return impl_conf > (meta_confidence + IMPLICIT_LOW_CONF_MARGIN)


async def request_implicit_emotion(
    redis,
    message_id: str,
    text: str,
    conv_history: list,
) -> Optional[dict]:
    """
    Publish an implicit emotion request and wait up to IMPLICIT_TIMEOUT_MS for
    a response from the llm_reasoning_service.
    """
    response_key = f"implicit_resp:{message_id}"
    timeout_secs = max(0.1, IMPLICIT_TIMEOUT_MS / 1000)

    try:
        await redis.xadd(REQUEST_STREAM, {
            "message_id":   message_id,
            "text":         text,
            "history":      json.dumps(conv_history[-3:] if conv_history else []),
            "response_key": response_key,
        })
    except Exception as e:
        logger.warning(f"[Implicit] Could not publish request for {message_id}: {e}")
        return None

    try:
        result = await redis.blpop([response_key], timeout=timeout_secs)
        if result:
            _, raw = result
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"[Implicit] BLPOP failed for {message_id}: {e}")

    return None


async def store_message_for_history(redis, conv_id: str, text: str) -> None:
    """Keep a rolling window of the last 3 message texts per conversation."""
    key = f"conv:{conv_id}:implicit_history"
    try:
        await redis.rpush(key, text)
        await redis.ltrim(key, -3, -1)
        await redis.expire(key, 3600)
    except Exception:
        pass


async def get_conv_history(redis, conv_id: str) -> list:
    """Retrieve the last ≤3 message texts for this conversation."""
    key = f"conv:{conv_id}:implicit_history"
    try:
        items = await redis.lrange(key, 0, -1)
        return [t.decode() if isinstance(t, bytes) else t for t in items]
    except Exception:
        return []
