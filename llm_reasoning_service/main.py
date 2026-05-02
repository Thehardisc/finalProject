import asyncio
import sys
import os
import json
import time

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger

logger = get_logger("llm_reasoning_service")

# --- Explainer Logic ---
class RuleBasedExplainer:
    """
    Generates human-readable explanations of emotion decisions.

    Currently uses a rule/template-based approach keyed on which model
    contributed most to the final decision (logic_map). This is deterministic
    and reproducible — ideal for a demo environment.

    To upgrade to a live LLM: set LLM_PROVIDER=OPENAI or LLM_PROVIDER=GROQ
    in .env and implement the API call in generate_insight().
    """
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

        if strongest == "EmojiNet":
            insights.append(f"Analysis is heavily driven by visual cues. The use of specific emojis confirms a strong {emotion} vibe.")
        elif strongest == "BERT":
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

explainer = RuleBasedExplainer()

async def main():
    await redis_client.connect()
    r = redis_client.redis
    
    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

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
                            # context_shift is a dict object if a shift was detected, None otherwise
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
            logger.log_exception("EXPLAINER SERVICE — Redis error, retrying in 1s", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
