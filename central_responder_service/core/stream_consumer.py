"""
core/stream_consumer.py — Redis stream consumer loop for the central responder.

Reads partial model results from partial_analysis_stream, accumulates them
per message until all 3 ML models have responded, then computes EmojiNet
inline (dict lookup — no network hop) and triggers aggregation.
"""
import json
import time

import emoji as emoji_lib

from shared.utils.logger import get_logger
from shared.utils.redis_client import RedisClient
from shared.constants import EMOJI_EMOTION_DB

from .aggregator import aggregate_and_publish

logger = get_logger("central_responder")

INPUT_STREAM       = "partial_analysis_stream"
GROUP_NAME         = "central_responder_group"
CONSUMER_NAME      = "responder_1"
PENDING_KEY_PREFIX = "pending_aggregation:"

# emojinet removed from the network hop — computed inline below
EXPECTED_MODELS = ["go_emotions", "basic_bert", "vader"]


def _emojinet_inline(text: str) -> dict:
    """
    Compute emoji emotion scores inline using the canonical EMOJI_EMOTION_DB.
    Replaces the entire emojinet_service Docker container.
    Returns a dict of {emotion: score} or {} when no known emoji are present.
    """
    try:
        found  = emoji_lib.distinct_emoji_list(text)
        scores = {}
        count  = 0
        for ch in found:
            entry = EMOJI_EMOTION_DB.get(ch) or EMOJI_EMOTION_DB.get(ch.replace('\ufe0f', ''))
            if entry:
                for emo, sc in entry.get("emotions", {}).items():
                    scores[emo] = scores.get(emo, 0.0) + sc
                count += 1
        if count:
            # Normalize + explicitly zero-out neutral to dilute other models
            result = {k: v / count for k, v in scores.items()}
            result.setdefault("neutral", 0.0)
            return result
    except Exception as e:
        logger.warning(f"Inline emojinet error: {e}")
    return {}


async def run_consumer_loop(redis_client: RedisClient, get_meta_learner) -> None:
    """
    Main event loop — blocks on xreadgroup and dispatches aggregation
    when all 3 model partials for a message have arrived.

    EmojiNet is now computed synchronously inline (microseconds) rather than
    waiting for a separate Docker service over Redis.

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

                        scores = scores_raw
                        if isinstance(scores_raw, str):
                            try:
                                scores = json.loads(scores_raw)
                            except Exception:
                                pass

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

                        is_new = not await r.exists(pending_key)
                        if is_new:
                            await r.hset(pending_key, "arrival_timestamp", time.time())
                            await r.expire(pending_key, 30)

                        await r.hset(pending_key, model_name, json.dumps(full_packet))

                        current_results = await r.hgetall(pending_key)
                        received_models = [k for k in current_results.keys()
                                           if k != "arrival_timestamp"]

                        if all(m in received_models for m in EXPECTED_MODELS):
                            logger.debug(f"All models received for {msg_id}. Aggregating.")

                            all_packets = []
                            arrival_ts  = float(current_results.get(
                                "arrival_timestamp", time.time()))
                            agg_lat = (time.time() - arrival_ts) * 1000

                            original_data = None
                            for m, packet_str in current_results.items():
                                if m == "arrival_timestamp":
                                    continue
                                try:
                                    pkt = json.loads(packet_str)
                                    all_packets.append(pkt)
                                    if original_data is None:
                                        original_data = pkt.get("original_data") or {}
                                except Exception:
                                    logger.error(f"Failed to parse packet for model {m}")

                            # ── Inline EmojiNet (replaces emojinet_service) ──────
                            text = original_data.get("text", "") if original_data else ""
                            emoji_scores = _emojinet_inline(text)
                            all_packets.append({
                                "model_name":    "emojinet",
                                "scores":        emoji_scores,
                                "original_data": original_data
                            })
                            logger.debug(
                                f"Inline emojinet: {len(emoji_scores)} emotion scores"
                                f" for msg {msg_id}"
                            )

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

                        await r.xack(INPUT_STREAM, GROUP_NAME, record_id)

        except Exception as e:
            logger.log_exception("CENTRAL RESPONDER CRITICAL ERROR", e)
            import asyncio
            await asyncio.sleep(1)
