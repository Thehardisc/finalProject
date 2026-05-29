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

async def main():
    await redis_client.connect()
    r = redis_client.redis
    
    # Create consumer group if not exists
    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    logger.info("Preprocessing worker started.")
    
    while True:
        try:
            streams = {STREAM_KEY: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=1, block=2000)

            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        mid  = data.get("message_id", "")
                        cid  = data.get("conversation_id", "")
                        uid  = data.get("user_id", "")
                        mlog = logger.bind(message_id=mid, conversation_id=cid, user_id=uid)
                        try:
                            mlog.debug("preprocess_start", extra={"event": "preprocess_start"})

                            original_text = data.get("text", "")
                            processed_text = clean_text(original_text)
                            processed_text_demojized = demojize_text(processed_text)

                            output_event = data.copy()
                            output_event["processed_text"] = processed_text
                            output_event["processed_text_demojized"] = processed_text_demojized
                            output_event["original_text"] = original_text

                            await redis_client.publish_event(OUTPUT_STREAM, output_event)

                            mlog.info(
                                "preprocess_done",
                                extra={
                                    "event":              "preprocess_done",
                                    "original_len":       len(original_text),
                                    "demojized_len":      len(processed_text_demojized),
                                },
                            )

                            await r.xack(STREAM_KEY, GROUP_NAME, message_id)

                        except Exception as msg_err:
                            mlog.error(
                                "preprocess_failed",
                                extra={
                                    "event":       "preprocess_failed",
                                    "error_class": type(msg_err).__name__,
                                    "error":       str(msg_err),
                                },
                            )
                            try:
                                await r.xack(STREAM_KEY, GROUP_NAME, message_id)
                            except Exception as ack_err:
                                mlog.warning("xack_failed", extra={"event": "xack_failed", "error": str(ack_err)})

        except Exception as e:
            logger.log_exception("PREPROCESSING WORKER — Redis error, retrying in 1s", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
