import asyncio
import sys
import os
import json
import time

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from implicit_detector import detect as detect_implicit_emotion

logger = get_logger("llm_reasoning_service")

# --- Explainer Logic ---
class RuleBasedExplainer:
    def __init__(self):
        self.provider_type = os.getenv("LLM_PROVIDER", "RULE_BASED").upper()
        logger.info(f"Explainer initialized in mode: {self.provider_type}")

    async def generate_insight(self, text, emotion, logic_map, sarcasm_score=0.0, conflict_desc=None, context_shift=False):
        # Rule-based explainer — deterministic and reproducible
        return self._rule_based_reasoning(text, emotion, logic_map, sarcasm_score, conflict_desc, context_shift)

    def _rule_based_reasoning(self, text, emotion, logic_map, sarcasm_score, conflict_desc, context_shift):
        if sarcasm_score > 0.5:
            return f"Detected high probability of sarcasm ({sarcasm_score:.0%}). {conflict_desc or 'Semantic signals are at odds with visual cues.'}"

        if not logic_map:
            return f"Analyzing '{text}' as {emotion}."

        strongest = max(logic_map.items(), key=lambda x: x[1])[0]
        insights = []

        if context_shift:
            insights.append(f"Detected a sudden emotional shift. While the text seems {emotion}, the conversation trajectory suggests underlying complexity.")

        if strongest == "BERT":
            insights.append(f"Deep linguistic patterns suggest a layer of {emotion} that isn't immediately obvious from keywords alone.")
        elif strongest == "VADER":
            insights.append(f"The sentiment is clearly defined by explicit emotional keywords in the message.")
        elif strongest == "GoEmotions":
            insights.append(f"The GoEmotions contextual model found strong signals for {emotion} within the full sentence structure.")
        elif strongest == "Context":
            insights.append(f"Current sentiment is heavily influenced by the previous tone of the conversation.")

        if not insights:
            return f"Predominant emotional state is {emotion}, verified across multiple analyzer models."

        return " ".join(insights[:2])

# --- Main Service Logic ---
redis_client = RedisClient()
STREAM_KEY = "conversation_update_stream"
GROUP_NAME = "reasoning_group"
CONSUMER_NAME = "reasoning_worker_1"
REASONING_UPDATE_KEY = "reasoning_update_stream"

# Implicit emotion request stream (Stage 0)
IMPLICIT_STREAM      = "implicit_emotion_requests"
IMPLICIT_GROUP       = "implicit_reasoning_group"
IMPLICIT_CONSUMER    = "implicit_worker_1"

explainer = RuleBasedExplainer()


async def run_explainer_loop(r):
    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating explainer group: {e}")

    logger.info("Cognitive Explainer Service ready.")

    while True:
        try:
            streams = {STREAM_KEY: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=1, block=5000)

            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        try:
                            text = data.get("original_text", "")
                            emotion = data.get("dominant_emotion", "Neutral")
                            pipeline_log = json.loads(data.get("pipeline_log", "{}"))
                            logic_map = pipeline_log.get("logic_map", {})
                            sarcasm_score = pipeline_log.get("sarcasm_score", 0.0)
                            conflict_desc = pipeline_log.get("conflict", None)
                            context_shift_raw = data.get("context_shift", "null")
                            try:
                                context_shift = json.loads(context_shift_raw)
                            except (json.JSONDecodeError, TypeError):
                                context_shift = None
                            is_context_shift = isinstance(context_shift, dict)
                            msg_uuid = data.get("message_id")

                            insight = await explainer.generate_insight(
                                text, emotion, logic_map,
                                sarcasm_score, conflict_desc, is_context_shift
                            )

                            payload = {
                                "message_id": msg_uuid,
                                "ai_insight": insight,
                                "timestamp": time.time()
                            }
                            await redis_client.publish_event(REASONING_UPDATE_KEY, payload)
                            await r.xack(STREAM_KEY, GROUP_NAME, message_id)

                        except Exception as msg_err:
                            logger.error(f"[EXPLAINER] Failed on {message_id}: {msg_err}. ACKing.")
                            try:
                                await r.xack(STREAM_KEY, GROUP_NAME, message_id)
                            except Exception:
                                pass

        except Exception as e:
            if "NOGROUP" in str(e):
                try:
                    await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
                    logger.info("Re-created explainer consumer group after NOGROUP error.")
                except Exception as cg_err:
                    if "BUSYGROUP" not in str(cg_err):
                        logger.error(f"Failed to re-create explainer group: {cg_err}")
            else:
                logger.log_exception("EXPLAINER SERVICE — Redis error, retrying in 1s", e)
            await asyncio.sleep(1)


async def run_implicit_loop(r):
    """Consumer loop for implicit_emotion_requests stream (Stage 0)."""
    try:
        await r.xgroup_create(IMPLICIT_STREAM, IMPLICIT_GROUP, id="0", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating implicit group: {e}")

    logger.info("Implicit emotion detector ready.")

    while True:
        try:
            streams  = {IMPLICIT_STREAM: ">"}
            messages = await r.xreadgroup(IMPLICIT_GROUP, IMPLICIT_CONSUMER, streams, count=4, block=2000)

            if messages:
                for stream, msgs in messages:
                    for entry_id, data in msgs:
                        try:
                            message_id   = (data.get("message_id") or b"").decode() if isinstance(data.get("message_id"), bytes) else str(data.get("message_id", ""))
                            text         = (data.get("text") or b"").decode() if isinstance(data.get("text"), bytes) else str(data.get("text", ""))
                            response_key = (data.get("response_key") or b"").decode() if isinstance(data.get("response_key"), bytes) else str(data.get("response_key", ""))

                            raw_hist = data.get("history", "[]")
                            if isinstance(raw_hist, bytes):
                                raw_hist = raw_hist.decode()
                            try:
                                history = json.loads(raw_hist)
                            except Exception:
                                history = []

                            result = detect_implicit_emotion(text, history)

                            if result is None:
                                result = {
                                    "emotion":    "neutral",
                                    "confidence": 0.0,
                                    "scores":     {"neutral": 1.0},
                                    "method":     "rule-based",
                                }

                            logger.debug(
                                f"[Implicit] {message_id}: → {result['emotion']} "
                                f"(conf={result['confidence']:.2f}, method={result['method']})"
                            )

                            # LPUSH so the BLPOP in central_responder unblocks
                            await r.lpush(response_key, json.dumps(result))
                            # Short TTL — if BLPOP already timed out the key should not linger
                            await r.expire(response_key, 5)
                            await r.xack(IMPLICIT_STREAM, IMPLICIT_GROUP, entry_id)

                        except Exception as msg_err:
                            logger.error(f"[Implicit] Failed on entry {entry_id}: {msg_err}. ACKing.")
                            try:
                                await r.xack(IMPLICIT_STREAM, IMPLICIT_GROUP, entry_id)
                            except Exception:
                                pass

        except Exception as e:
            if "NOGROUP" in str(e):
                try:
                    await r.xgroup_create(IMPLICIT_STREAM, IMPLICIT_GROUP, id="0", mkstream=True)
                    logger.info("Re-created implicit consumer group after NOGROUP error.")
                except Exception as cg_err:
                    if "BUSYGROUP" not in str(cg_err):
                        logger.error(f"Failed to re-create implicit group: {cg_err}")
            else:
                logger.log_exception("IMPLICIT LOOP — Redis error, retrying in 1s", e)
            await asyncio.sleep(1)


async def main():
    await redis_client.connect()
    r = redis_client.redis

    await asyncio.gather(
        run_explainer_loop(r),
        run_implicit_loop(r),
    )


if __name__ == "__main__":
    asyncio.run(main())
