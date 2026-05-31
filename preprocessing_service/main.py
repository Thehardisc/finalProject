import asyncio
import sys
import os
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from preprocessing_service.utils import clean_text, demojize_text
from preprocessing_service.debouncer import MessageDebouncer, DEBOUNCE_WINDOW_MS

logger = get_logger("preprocessing_service")

redis_client  = RedisClient()
STREAM_KEY    = "message_stream"
GROUP_NAME    = "preprocessing_group"
CONSUMER_NAME = "worker_1"
OUTPUT_STREAM = "preprocessed_stream"


async def _process_one(
    r,
    record_id:       str,
    data:            dict,
    is_continuation: bool = False,
    burst_fragments: str = None,
) -> None:
    """
    Clean, demojize, tag, and forward one message to preprocessed_stream.

    is_continuation=True means this message is a rapid-fire fragment of the
    same thought as the preceding message (same user, same conversation,
    within DEBOUNCE_WINDOW_MS).  Downstream services — specifically the
    Context Engine — use this flag to suppress CDM state transitions and
    velocity spikes for what is semantically a single conversational turn.
    """
    try:
        original_text            = data.get("text", "")
        processed_text           = clean_text(original_text)
        processed_text_demojized = demojize_text(processed_text)

        output_event = data.copy()
        output_event["processed_text"]           = processed_text
        output_event["processed_text_demojized"] = processed_text_demojized
        output_event["original_text"]            = original_text
        output_event["is_continuation"]          = "true" if is_continuation else "false"
        if burst_fragments:
            output_event["burst_fragments"]      = burst_fragments

        await redis_client.publish_event(OUTPUT_STREAM, output_event)

        logger.log_stats("PREPROCESSING COMPLETE", {
            "Message ID":      data.get("message_id", "N/A"),
            "is_continuation": is_continuation,
            "Original":        (original_text[:40] + "...") if len(original_text) > 40 else original_text,
            "Demojized":       (processed_text_demojized[:40] + "...") if len(processed_text_demojized) > 40 else processed_text_demojized,
        })
    except Exception as msg_err:
        logger.error(
            f"[PREPROCESSING] Failed on {record_id}: {msg_err}. ACKing to prevent requeue."
        )
    finally:
        try:
            await r.xack(STREAM_KEY, GROUP_NAME, record_id)
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

    debouncer = MessageDebouncer()
    logger.info(
        f"Preprocessing worker started.  "
        f"Debounce window={DEBOUNCE_WINDOW_MS}ms"
    )

    async def _flush_burst(
        items: list,
        flags: list,
    ) -> None:
        """
        Process and ACK a flushed burst.

        items : [(record_id, data), ...]
        flags : [is_continuation, ...]
        """
        if len(items) > 1:
            user  = items[0][1].get("user_id",          "?")
            conv  = items[0][1].get("conversation_id",  "?")
            logger.info(
                f"[Debouncer] Burst flush: {len(items)} messages  "
                f"user={user}  conv={conv}  "
                f"continuations={sum(flags)}"
            )
        if len(items) > 1:
            combined_text = " ".join(data.get("text", "") for _, data in items)
            fragments_data = [{"message_id": data.get("message_id")} for _, data in items]
            
            # Use the first item's metadata as the base payload
            first_record_id, first_data = items[0]
            first_data = first_data.copy()
            first_data["text"] = combined_text
            
            import json
            await _process_one(
                r, first_record_id, first_data,
                is_continuation=flags[0],
                burst_fragments=json.dumps(fragments_data)
            )
        else:
            first_record_id, first_data = items[0]
            await _process_one(r, first_record_id, first_data, is_continuation=flags[0])

    while True:
        try:
            # PEL recovery: reclaim records idle >30 s (crash-before-ACK protection).
            # Reclaimed messages are treated as fresh turns (is_continuation=False)
            # because burst context is lost on restart — safe, just slightly
            # less accurate for the rare mid-burst crash case.
            try:
                _, stale, _ = await r.xautoclaim(
                    STREAM_KEY, GROUP_NAME, CONSUMER_NAME,
                    min_idle_time=30_000, start_id="0-0", count=10,
                )
                for record_id, data in (stale or []):
                    logger.info(f"[PEL] Reclaimed stale message {record_id}")
                    await _process_one(r, record_id, data, is_continuation=False)
            except Exception:
                pass

            # count=10: read a full batch so bursts arriving simultaneously
            # are all added to the debouncer in the same loop iteration.
            # block=500: short enough that debounce timers fire promptly while
            # idle; asyncio yields freely during the await, so all in-flight
            # tasks continue running during the server-side wait.
            messages = await r.xreadgroup(
                GROUP_NAME, CONSUMER_NAME,
                {STREAM_KEY: ">"},
                count=10, block=500,
            )
            if messages:
                for stream, msgs in messages:
                    for record_id, data in msgs:
                        await debouncer.add(record_id, data, _flush_burst)

        except Exception as e:
            if "NOGROUP" in str(e):
                try:
                    await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
                    logger.info("Re-created consumer group after NOGROUP error.")
                except Exception as cg_err:
                    if "BUSYGROUP" not in str(cg_err):
                        logger.error(f"Failed to re-create group: {cg_err}")
            else:
                logger.log_exception(
                    "PREPROCESSING WORKER — Redis error, retrying in 1s", e
                )
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
