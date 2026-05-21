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
                to_ack: list = []
                to_dlq: list = []  # R2: events that fail processing
                try:
                    for stream, msgs in messages:
                        for message_id, data in msgs:
                            logger.debug(f"Persisting data from stream: {stream}")

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

                            to_ack.append((stream, message_id))

                    session.commit()

                    for ack_stream, ack_id in to_ack:
                        await r.xack(ack_stream, "persistence_group", ack_id)

                    elapsed = (time.time() - start_time) * 1000
                    logger.info(
                        f"Successfully persisted batch of {len(to_ack)} events "
                        f"in {elapsed:.2f}ms"
                    )

                except Exception as e:
                    session.rollback()
                    logger.log_exception(
                        "SQL TRANSACTION ROLLBACK — routing to DLQ", e
                    )
                    # R2: send failed events to DLQ instead of silently dropping
                    for dlq_stream, dlq_id in to_dlq:
                        try:
                            await r.xadd(DLQ_STREAM, {
                                "original_stream": dlq_stream,
                                "original_id":     dlq_id,
                                "error":           str(e)[:500],
                                "timestamp":       time.time()
                            }, maxlen=5000, approximate=True)
                        except Exception:
                            pass
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
