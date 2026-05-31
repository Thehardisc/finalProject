"""
aggregation_service/main.py — Entry point for the aggregation worker.

Reads from the emotion_stream, updates conversation state via Redis,
and publishes the enriched event to the conversation_update_stream.
"""
import asyncio
import json
import os
import signal
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from aggregation_service.state.conversation_state import update_conversation_state

logger = get_logger("aggregation_service")

redis_client  = RedisClient()
STREAM_KEY    = "emotion_stream"
GROUP_NAME    = "aggregation_group"
CONSUMER_NAME = "worker_1"
OUTPUT_STREAM = "conversation_update_stream"

_shutdown = False  # R1


async def main():
    await redis_client.connect()
    r = redis_client.redis

    # R1: graceful shutdown on SIGTERM
    loop = asyncio.get_event_loop()
    def _handle_sigterm():
        global _shutdown
        print("[aggregation_service] SIGTERM — finishing current event then exiting.", flush=True)
        _shutdown = True
    loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)

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
                        mid  = data.get("message_id", "")
                        cid  = data.get("conversation_id", "")
                        uid  = data.get("user_id", "")
                        mlog = logger.bind(message_id=mid, conversation_id=cid, user_id=uid)
                        try:
                            conversation_id = data.get("conversation_id")
                            original_text   = data.get("original_text", "")
                            user_id         = data.get("user_id", "")

                            emotions_map = {}
                            try:
                                emotions_map = json.loads(data.get("emotions", "{}"))
                            except json.JSONDecodeError as e:
                                mlog.warning(
                                    "emotions_parse_failed",
                                    extra={"event": "emotions_parse_failed", "error": str(e)},
                                )

                            dom_emo = data.get("dominant_emotion")
                            if dom_emo:
                                emotions_map["dominant_emotion"] = dom_emo

                            if conversation_id:
                                t0 = time.time()
                                updated_state = await update_conversation_state(
                                    conversation_id, new_emotions=emotions_map,
                                    r=r, original_text=original_text,
                                    user_id=user_id,
                                )
                                elapsed = (time.time() - t0) * 1000
                                mlog.info(
                                    "aggregation_done",
                                    extra={
                                        "event":      "aggregation_done",
                                        "latency_ms": round(elapsed, 2),
                                        "dominant":   dom_emo or "unknown",
                                        "avg_val":    round(float(updated_state.get("average_valence", 0.0)), 4) if updated_state else 0.0,
                                    },
                                )
                            else:
                                mlog.warning("no_conversation_id", extra={"event": "no_conversation_id"})
                                updated_state = {}

                            output_event = data.copy()
                            output_event["conversation_state"] = json.dumps(updated_state)
                            
                            burst_fragments_str = data.get("burst_fragments")
                            if burst_fragments_str:
                                try:
                                    fragments = json.loads(burst_fragments_str)
                                    for frag in fragments:
                                        frag_event = output_event.copy()
                                        frag_event["message_id"] = frag["message_id"]
                                        frag_event.pop("burst_fragments", None)
                                        await redis_client.publish_event(OUTPUT_STREAM, frag_event)
                                except Exception as e:
                                    logger.error(f"Failed to unpack burst fragments for {message_id}: {e}")
                                    await redis_client.publish_event(OUTPUT_STREAM, output_event)
                            else:
                                await redis_client.publish_event(OUTPUT_STREAM, output_event)
                                
                            await r.xack(STREAM_KEY, GROUP_NAME, message_id)

                        except Exception as msg_err:
                            mlog.error(
                                "aggregation_failed",
                                extra={
                                    "event":       "aggregation_failed",
                                    "error_class": type(msg_err).__name__,
                                    "error":       str(msg_err),
                                },
                            )
                            try:
                                await r.xack(STREAM_KEY, GROUP_NAME, message_id)
                            except Exception as ack_err:
                                mlog.warning("xack_failed", extra={"event": "xack_failed", "error": str(ack_err)})

        except Exception as e:
            logger.log_exception(
                "AGGREGATION WORKER — Redis connection error, retrying in 2s", e
            )
            await asyncio.sleep(2)

        if _shutdown:  # R1
            logger.info("Aggregation service: graceful shutdown complete.")
            break


if __name__ == "__main__":
    asyncio.run(main())
