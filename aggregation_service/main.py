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
from aggregation_service.emotion_dynamics import compute_inertia, compute_contagion

logger = get_logger("aggregation_service")

redis_client  = RedisClient()
STREAM_KEY    = "emotion_stream"
GROUP_NAME    = "aggregation_group"
CONSUMER_NAME = "worker_1"
OUTPUT_STREAM = "conversation_update_stream"

_shutdown = False


async def main():
    await redis_client.connect()
    r = redis_client.redis

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
            try:
                _, stale, _ = await r.xautoclaim(
                    STREAM_KEY, GROUP_NAME, CONSUMER_NAME,
                    min_idle_time=30_000, start_id="0-0", count=5,
                )
                if stale:
                    logger.info(f"[PEL] Reclaimed {len(stale)} stale message(s)")
                    messages = [(STREAM_KEY, stale)]
                else:
                    messages = await r.xreadgroup(
                        GROUP_NAME, CONSUMER_NAME,
                        {STREAM_KEY: ">"}, count=1, block=2000,
                    )
            except Exception:
                messages = await r.xreadgroup(
                    GROUP_NAME, CONSUMER_NAME,
                    {STREAM_KEY: ">"}, count=1, block=2000,
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

                            # vader_compound is in the "vader" key, not in "emotions" (pure 28-class GoE)
                            try:
                                vader_parsed = json.loads(data.get("vader", "{}"))
                                emotions_map["vader_compound"] = float(vader_parsed.get("vader_compound", 0.0))
                            except Exception:
                                emotions_map.setdefault("vader_compound", 0.0)

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

                                try:
                                    vader_data = {}
                                    try:
                                        vader_data = json.loads(data.get("vader", "{}"))
                                    except Exception:
                                        pass
                                    msg_valence = float(vader_data.get("vader_compound", 0.0))

                                    spk_key = f"conv:{conversation_id}:spk:{user_id}:valence_seq"
                                    raw_seq = await r.get(spk_key)
                                    spk_seq = json.loads(raw_seq) if raw_seq else []
                                    spk_seq.append(msg_valence)
                                    if len(spk_seq) > 20:
                                        spk_seq = spk_seq[-20:]
                                    await r.set(spk_key, json.dumps(spk_seq), ex=3600)

                                    inertia = compute_inertia(spk_seq)
                                    await r.set(
                                        f"conv:{conversation_id}:inertia:{user_id}",
                                        str(inertia), ex=3600,
                                    )

                                    pattern = f"conv:{conversation_id}:spk:*:valence_seq"
                                    other_keys = [
                                        k async for k in r.scan_iter(pattern, count=100)
                                        if f":spk:{user_id}:" not in k
                                    ]
                                    if other_keys:
                                        raw_other = await r.get(other_keys[0])
                                        other_seq = json.loads(raw_other) if raw_other else []
                                        contagion = compute_contagion(spk_seq, other_seq)
                                    else:
                                        contagion = 0.0
                                    await r.set(
                                        f"conv:{conversation_id}:contagion",
                                        str(contagion), ex=3600,
                                    )
                                    mlog.debug(
                                        "dynamics_computed",
                                        extra={
                                            "event":     "dynamics_computed",
                                            "inertia":   round(inertia, 4),
                                            "contagion": round(contagion, 4),
                                        },
                                    )
                                except Exception as _dyn_err:
                                    mlog.warning(
                                        "dynamics_failed",
                                        extra={"event": "dynamics_failed", "error": str(_dyn_err)},
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
            if "NOGROUP" in str(e):
                try:
                    await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
                    logger.warning("Re-created consumer group after NOGROUP error.")
                except Exception as cg_err:
                    if "BUSYGROUP" not in str(cg_err):
                        logger.error(f"Failed to re-create group: {cg_err}")
            else:
                logger.log_exception(
                    "AGGREGATION WORKER — Redis connection error, retrying in 2s", e
                )
            await asyncio.sleep(2)

        if _shutdown:
            logger.info("Aggregation service: graceful shutdown complete.")
            break


if __name__ == "__main__":
    asyncio.run(main())
