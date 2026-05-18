"""
aggregation_service/main.py — Entry point for the aggregation worker.

Reads from the emotion_stream, updates conversation state via Redis,
and publishes the enriched event to the conversation_update_stream.
"""
import asyncio
import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from aggregation_service.state.conversation_state import update_conversation_state

logger = get_logger("aggregation_service")

redis_client = RedisClient()
STREAM_KEY   = "emotion_stream"
GROUP_NAME   = "aggregation_group"
CONSUMER_NAME = "worker_1"
OUTPUT_STREAM = "conversation_update_stream"


async def main():
    await redis_client.connect()
    r = redis_client.redis

    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    logger.info("Aggregation worker started.")

    while True:
        try:
            messages = await r.xreadgroup(
                GROUP_NAME, CONSUMER_NAME,
                {STREAM_KEY: ">"}, count=1, block=2000
            )

            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        try:
                            conversation_id = data.get("conversation_id")
                            original_text   = data.get("original_text", "")

                            emotions_map = {}
                            try:
                                emotions_map = json.loads(data.get("emotions", "{}"))
                            except json.JSONDecodeError:
                                pass

                            dom_emo = data.get("dominant_emotion")
                            if dom_emo:
                                emotions_map["dominant_emotion"] = dom_emo

                            t0 = time.time()
                            updated_state = await update_conversation_state(
                                conversation_id, new_emotions=emotions_map,
                                r=r, original_text=original_text
                            )
                            elapsed = (time.time() - t0) * 1000
                            logger.debug(
                                f"Aggregation for {conversation_id} took {elapsed:.2f}ms"
                            )

                            output_event = data.copy()
                            output_event["conversation_state"] = json.dumps(updated_state)
                            await redis_client.publish_event(OUTPUT_STREAM, output_event)
                            await r.xack(STREAM_KEY, GROUP_NAME, message_id)

                        except Exception as msg_err:
                            logger.error(
                                f"[AGGREGATION] Failed to process {message_id}: {msg_err}. "
                                "ACKing to prevent infinite requeue."
                            )
                            try:
                                await r.xack(STREAM_KEY, GROUP_NAME, message_id)
                            except Exception:
                                pass

        except Exception as e:
            logger.log_exception(
                "AGGREGATION WORKER — Redis connection error, retrying in 2s", e
            )
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
