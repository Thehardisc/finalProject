"""
core/stream_consumer.py — Redis stream consumer loop for the central responder.

Reads partial model results from the partial_analysis_stream, accumulates them
per message until all 4 models have responded, then triggers aggregation.
"""
import json
import time

from shared.utils.logger import get_logger
from shared.utils.redis_client import RedisClient

from .aggregator import aggregate_and_publish

logger = get_logger("central_responder")

INPUT_STREAM    = "partial_analysis_stream"
GROUP_NAME      = "central_responder_group"
CONSUMER_NAME   = "responder_1"
PENDING_KEY_PREFIX = "pending_aggregation:"
EXPECTED_MODELS = ["go_emotions", "basic_bert", "vader", "emojinet"]


async def run_consumer_loop(redis_client: RedisClient, get_meta_learner) -> None:
    """
    Main event loop — blocks on xreadgroup and dispatches aggregation
    when all 4 model partials for a message have arrived.

    get_meta_learner is a callable that returns the current META_LEARNER
    (supports hot-reload without restarting the loop).
    """
    r = redis_client.redis

    try:
        await r.xgroup_create(INPUT_STREAM, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating consumer group: {e}")

    logger.info("Central Responder consumer loop started.")

    while True:
        try:
            streams  = {INPUT_STREAM: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams,
                                          count=10, block=2000)

            if messages:
                for stream, msgs in messages:
                    for record_id, data in msgs:
                        msg_id     = data.get("message_id")
                        model_name = data.get("model_name")
                        scores_raw = data.get("scores")

                        # Deserialise scores if they arrived as JSON string
                        scores = scores_raw
                        if isinstance(scores_raw, str):
                            try:
                                scores = json.loads(scores_raw)
                            except Exception:
                                pass

                        # Build the pending packet for this model
                        full_packet = {
                            "model_name":    model_name,
                            "scores":        scores,
                            "original_data": data.get("original_data")
                        }
                        if isinstance(full_packet["original_data"], str):
                            try:
                                full_packet["original_data"] = json.loads(
                                    full_packet["original_data"])
                            except Exception:
                                pass

                        pending_key = f"{PENDING_KEY_PREFIX}{msg_id}"

                        # Stamp arrival time and set TTL on first packet
                        is_new = not await r.exists(pending_key)
                        if is_new:
                            await r.hset(pending_key, "arrival_timestamp", time.time())
                            await r.expire(pending_key, 30)  # zombie-key guard

                        await r.hset(pending_key, model_name, json.dumps(full_packet))

                        # Check if all models have arrived
                        current_results  = await r.hgetall(pending_key)
                        received_models  = [k for k in current_results.keys()
                                            if k != "arrival_timestamp"]

                        if all(m in received_models for m in EXPECTED_MODELS):
                            logger.debug(f"All models received for {msg_id}. Aggregating.")

                            all_packets = []
                            arrival_ts  = float(current_results.get(
                                "arrival_timestamp", time.time()))
                            agg_lat = (time.time() - arrival_ts) * 1000

                            for m, packet_str in current_results.items():
                                if m == "arrival_timestamp":
                                    continue
                                try:
                                    all_packets.append(json.loads(packet_str))
                                except Exception:
                                    logger.error(f"Failed to parse packet for model {m}")

                            try:
                                await aggregate_and_publish(
                                    msg_id, all_packets,
                                    get_meta_learner(), redis_client,
                                    r, agg_lat=agg_lat
                                )
                            except Exception as agg_err:
                                logger.error(f"[AGGREGATION FAILED] {msg_id}: {agg_err}")
                        else:
                            missing = list(set(EXPECTED_MODELS) - set(received_models))
                            logger.debug(f"Message {msg_id} waiting for: {missing}")

                        # Always ACK so the stream doesn't fill up
                        await r.xack(INPUT_STREAM, GROUP_NAME, record_id)

        except Exception as e:
            logger.log_exception("CENTRAL RESPONDER CRITICAL ERROR", e)
            import asyncio
            await asyncio.sleep(1)
