import asyncio
import sys
import os
import json
import time

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from preprocessing_service.utils import clean_text, demojize_text

logger = get_logger("preprocessing_service")

redis_client = RedisClient()
STREAM_KEY = "message_stream"
GROUP_NAME = "preprocessing_group"
CONSUMER_NAME = "worker_1"
OUTPUT_STREAM = "preprocessed_stream"

async def _process_one(r, message_id, data):
    try:
        original_text = data.get("text", "")
        processed_text = clean_text(original_text)
        processed_text_demojized = demojize_text(processed_text)

        output_event = data.copy()
        output_event["processed_text"] = processed_text
        output_event["processed_text_demojized"] = processed_text_demojized
        output_event["original_text"] = original_text

        await redis_client.publish_event(OUTPUT_STREAM, output_event)

        logger.log_stats("PREPROCESSING COMPLETE", {
            "Message ID": data.get("message_id", "N/A"),
            "Original":   (original_text[:40] + '...') if len(original_text) > 40 else original_text,
            "Demojized":  (processed_text_demojized[:40] + '...') if len(processed_text_demojized) > 40 else processed_text_demojized,
        })
    except Exception as msg_err:
        logger.error(f"[PREPROCESSING] Failed on {message_id}: {msg_err}. ACKing to prevent requeue.")
    finally:
        try:
            await r.xack(STREAM_KEY, GROUP_NAME, message_id)
        except Exception:
            pass


async def main():
    await redis_client.connect()
    r = redis_client.redis

    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    logger.info("Preprocessing worker started.")

    while True:
        try:
            # PEL recovery: reclaim messages idle >30s (handles crash-before-ACK)
            try:
                _, stale, _ = await r.xautoclaim(STREAM_KEY, GROUP_NAME, CONSUMER_NAME, min_idle_time=30_000, start_id='0-0', count=10)
                for message_id, data in (stale or []):
                    logger.info(f"[PEL] Reclaimed stale message {message_id}")
                    await _process_one(r, message_id, data)
            except Exception:
                pass

            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, {STREAM_KEY: ">"}, count=1, block=2000)
            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        await _process_one(r, message_id, data)

        except Exception as e:
            if "NOGROUP" in str(e):
                try:
                    await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
                    logger.info("Re-created consumer group after NOGROUP error.")
                except Exception as cg_err:
                    if "BUSYGROUP" not in str(cg_err):
                        logger.error(f"Failed to re-create group: {cg_err}")
            else:
                logger.log_exception("PREPROCESSING WORKER — Redis error, retrying in 1s", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
