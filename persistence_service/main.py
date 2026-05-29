"""
persistence_service/main.py — Entry point for the persistence worker.

Reads from 4 Redis streams and dispatches each event type to its dedicated handler.
"""
import asyncio
import sys
import os
import signal
import time
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from persistence_service.db.setup import init_db
from persistence_service.handlers.message_handler  import process_message_event
from persistence_service.handlers.emotion_handler  import process_emotion_event
from persistence_service.handlers.state_handler    import process_state_event
from persistence_service.handlers.feedback_handler import process_feedback_event

print("[persistence_service] Module imports OK", flush=True)
logger = get_logger("persistence_service")

DB_USER     = os.getenv("POSTGRES_USER",    "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD","password")
DB_NAME     = os.getenv("POSTGRES_DB",      "emotion_db")
DB_HOST     = os.getenv("DB_HOST",          "db")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

redis_client = RedisClient()

_shutdown = False   # R1: set True on SIGTERM so the loop exits cleanly
DLQ_STREAM = "failed_events_stream"  # R2: dead letter queue

STREAMS = {
    "message_stream":             "persistence_group",
    "emotion_stream":             "persistence_group",
    "conversation_update_stream": "persistence_group",
    "feedback_stream":            "persistence_group"
}
CONSUMER_NAME = "worker_1"


async def main():
    print("[persistence_service] main() loop starting", flush=True)

    # R1: register SIGTERM so Docker stop finishes the current batch gracefully
    loop = asyncio.get_event_loop()
    def _handle_sigterm():
        global _shutdown
        print("[persistence_service] SIGTERM received — finishing current batch then exiting.", flush=True)
        _shutdown = True
    loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)

    for _ in range(10):
        try:
            init_db(engine)
            break
        except Exception:
            logger.info("Waiting for Database to be ready...")
            time.sleep(2)

    await redis_client.connect()
    r = redis_client.redis

    for stream, group in STREAMS.items():
        try:
            await r.xgroup_create(stream, group, mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Error creating group for {stream}: {e}")

    logger.info("Persistence worker started.")

    while True:
        try:
            streams_dict = {stream: ">" for stream in STREAMS}
            messages = await r.xreadgroup(
                "persistence_group", CONSUMER_NAME,
                streams_dict, count=10, block=2000
            )

            if messages:
                start_time = time.time()
                session    = SessionLocal()
                # R2/replay: track (stream, record_id, business_message_id) per event so DLQ
                # entries carry enough context to be replayable by message_id, not just record id.
                to_ack: list = []     # [(stream, record_id), ...]
                to_meta: list = []    # parallel list of business message_ids
                try:
                    for stream, msgs in messages:
                        for record_id, data in msgs:
                            biz_mid = data.get("message_id", "")
                            cid     = data.get("conversation_id", "")
                            uid     = data.get("user_id", "")
                            mlog = logger.bind(
                                message_id=biz_mid,
                                conversation_id=cid,
                                user_id=uid,
                                stream=stream,
                            )
                            mlog.debug("persist_dispatch", extra={"event": "persist_dispatch"})

                            if stream == "message_stream":
                                # Pv2: strip null bytes that corrupt PostgreSQL text columns
                                if "text" in data:
                                    data["text"] = data["text"].replace("\x00", "")
                                await process_message_event(session, data)
                            elif stream == "emotion_stream":
                                await process_emotion_event(session, data)
                            elif stream == "conversation_update_stream":
                                await process_state_event(session, data)
                            elif stream == "feedback_stream":
                                await process_feedback_event(session, data)

                            to_ack.append((stream, record_id))
                            to_meta.append(biz_mid)

                    session.commit()

                    for ack_stream, ack_id in to_ack:
                        await r.xack(ack_stream, "persistence_group", ack_id)

                    elapsed = (time.time() - start_time) * 1000
                    logger.info(
                        "persist_batch_done",
                        extra={
                            "event":      "persist_batch_done",
                            "batch_size": len(to_ack),
                            "latency_ms": round(elapsed, 2),
                        },
                    )

                except Exception as e:
                    session.rollback()
                    # Structured failure log so an operator can find the affected message_ids
                    # without parsing free-form text.
                    streams_in_batch = sorted({s for s, _ in to_ack})
                    logger.error(
                        "persistence_failed",
                        extra={
                            "event":       "persistence_failed",
                            "error_class": type(e).__name__,
                            "error":       str(e)[:500],
                            "batch_size":  len(to_ack),
                            "streams":     streams_in_batch,
                            "message_ids": [m for m in to_meta if m][:50],  # cap to avoid huge lines
                        },
                        exc_info=True,
                    )
                    # R2: send failed events to DLQ — now WITH the business message_id so they
                    # can actually be replayed (record_id alone is opaque once the stream rotates).
                    for (dlq_stream, dlq_id), biz_mid in zip(to_ack, to_meta):
                        try:
                            await r.xadd(DLQ_STREAM, {
                                "original_stream": dlq_stream,
                                "original_id":     dlq_id,
                                "message_id":      biz_mid or "",
                                "error_class":     type(e).__name__,
                                "error":           str(e)[:500],
                                "batch_size":      len(to_ack),
                                "timestamp":       time.time(),
                            }, maxlen=5000, approximate=True)
                        except Exception as dlq_err:
                            logger.warning(
                                "dlq_write_failed",
                                extra={
                                    "event":       "dlq_write_failed",
                                    "stream":      dlq_stream,
                                    "record_id":   dlq_id,
                                    "message_id":  biz_mid,
                                    "error":       str(dlq_err),
                                },
                            )
                finally:
                    session.close()

        except Exception as e:
            logger.log_exception("PERSISTENCE WORKER FATAL ERROR", e)
            await asyncio.sleep(1)

        # R1: exit cleanly after finishing a batch
        if _shutdown:
            logger.info("Graceful shutdown complete.")
            break


if __name__ == "__main__":
    try:
        print("[persistence_service] Starting main()...", flush=True)
        asyncio.run(main())
    except BaseException as e:
        print(f"[persistence_service] FATAL CRASH: {type(e).__name__} - {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
