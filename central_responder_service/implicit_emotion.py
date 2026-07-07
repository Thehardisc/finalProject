import json
import os


from shared.utils.logger import get_logger

logger = get_logger("implicit_emotion")

IMPLICIT_EMOTION_ENABLED     = os.environ.get("IMPLICIT_EMOTION_ENABLED", "true").lower() == "true"
IMPLICIT_TIMEOUT_MS          = int(os.environ.get("IMPLICIT_TIMEOUT_MS", "300"))
IMPLICIT_MIN_WORDS           = int(os.environ.get("IMPLICIT_MIN_WORDS", "4"))
IMPLICIT_GOE_NEUTRAL_CAP     = float(os.environ.get("IMPLICIT_GOE_NEUTRAL_CAP", "0.90"))
IMPLICIT_LOW_CONF_THRESHOLD  = float(os.environ.get("IMPLICIT_LOW_CONF_THRESHOLD", "0.42"))
IMPLICIT_LOW_CONF_MARGIN     = float(os.environ.get("IMPLICIT_LOW_CONF_MARGIN", "0.10"))
IMPLICIT_CONF_THRESHOLD      = float(os.environ.get("IMPLICIT_CONF_THRESHOLD", "0.50"))

REQUEST_STREAM = "implicit_emotion_requests"


def is_implicit_candidate(dominant_emotion, goe_scores, text, meta_confidence=1.0):
    if not IMPLICIT_EMOTION_ENABLED:
        return False
    if len(text.split()) < IMPLICIT_MIN_WORDS:
        return False
    if dominant_emotion == "neutral":
        return float(goe_scores.get("neutral", 0.0)) <= IMPLICIT_GOE_NEUTRAL_CAP
    return meta_confidence < IMPLICIT_LOW_CONF_THRESHOLD


_POSITIVE_FAMILY = {"joy", "amusement", "excitement", "admiration", "approval", "caring",
                    "gratitude", "love", "optimism", "pride", "relief", "desire"}
_NEGATIVE_FAMILY = {"anger", "annoyance", "disappointment", "disapproval", "disgust",
                    "sadness", "grief", "remorse", "fear", "nervousness", "embarrassment"}
IMPLICIT_FLIP_MARGIN = float(os.environ.get("IMPLICIT_FLIP_MARGIN", "0.30"))


def should_override(dominant_emotion, meta_confidence, impl_result,
                    goe_scores=None, vader_compound=0.0):
    impl_conf = float(impl_result.get("confidence", 0))
    impl_emo  = impl_result.get("emotion", "neutral")
    if impl_emo == "neutral":
        return False
    # Valence veto: the keyword detector's fixed confidences must never flip the
    # polarity of a message the text models read clearly ("literally shaking rn"
    # is not fear when GoE says excitement and VADER reads positive).
    if goe_scores:
        pos_mass = sum(float(goe_scores.get(e, 0.0)) for e in _POSITIVE_FAMILY)
        neg_mass = sum(float(goe_scores.get(e, 0.0)) for e in _NEGATIVE_FAMILY)
        if impl_emo in _NEGATIVE_FAMILY and (pos_mass > 0.5 or vader_compound >= 0.3):
            logger.info(f"[Implicit] veto: '{impl_emo}' vs positive text signal "
                        f"(goe_pos={pos_mass:.2f}, vader={vader_compound:+.2f})")
            return False
        if impl_emo in _POSITIVE_FAMILY and (neg_mass > 0.5 or vader_compound <= -0.3):
            logger.info(f"[Implicit] veto: '{impl_emo}' vs negative text signal "
                        f"(goe_neg={neg_mass:.2f}, vader={vader_compound:+.2f})")
            return False
    if dominant_emotion == "neutral":
        return impl_conf > IMPLICIT_CONF_THRESHOLD
    # Flipping valence polarity is drastic — demand a much larger margin than a
    # same-family refinement.
    flips = ((dominant_emotion in _POSITIVE_FAMILY and impl_emo in _NEGATIVE_FAMILY)
             or (dominant_emotion in _NEGATIVE_FAMILY and impl_emo in _POSITIVE_FAMILY))
    margin = IMPLICIT_FLIP_MARGIN if flips else IMPLICIT_LOW_CONF_MARGIN
    return impl_conf > (meta_confidence + margin)


async def request_implicit_emotion(redis, message_id, text, conv_history):
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
    finally:
        try:
            await redis.delete(response_key)
        except Exception:
            pass
    return None


async def store_message_for_history(redis, conv_id, text):
    key = f"conv:{conv_id}:implicit_history"
    try:
        await redis.rpush(key, text)
        await redis.ltrim(key, -3, -1)
        await redis.expire(key, 3600)
    except Exception:
        pass


async def get_conv_history(redis, conv_id):
    key = f"conv:{conv_id}:implicit_history"
    try:
        items = await redis.lrange(key, 0, -1)
        return [t.decode() if isinstance(t, bytes) else t for t in items]
    except Exception:
        return []
