"""
core/stream_consumer.py — Redis stream consumer loop for the central responder.

Reads partial model results from partial_analysis_stream, accumulates them per
message until all 3 ML models have responded, then triggers aggregation. A
background sweeper detects messages whose models never all arrive and routes them
to a dead-letter stream instead of letting them silently expire.
"""
import asyncio
import json
import time

from shared.utils.logger import get_logger
from shared.utils.redis_client import RedisClient

from core.aggregator import aggregate_and_publish

logger = get_logger("central_responder")

INPUT_STREAM       = "partial_analysis_stream"
GROUP_NAME         = "central_responder_group"
CONSUMER_NAME      = "responder_1"
PENDING_KEY_PREFIX = "pending_aggregation:"
DLQ_STREAM         = "failed_aggregation_stream"

EXPECTED_MODELS = ["go_emotions", "basic_bert", "vader"]

# A message whose 3 ML partials haven't all arrived within this window is considered
# stuck (a model crashed or never published) and routed to the DLQ. Kept well under
# the 30s pending-key TTL so the sweeper acts before Redis silently drops the key.
AGGREGATION_TIMEOUT_MS = 15000
SWEEP_INTERVAL_SECONDS = 5


async def _write_dlq(r, message_id: str, reason: str, error_class: str,
                     error: str, extra: dict = None) -> None:
    """Best-effort dead-letter write (mirrors persistence_service's DLQ pattern)."""
    entry = {
        "message_id":  message_id or "",
        "reason":      reason,
        "error_class": error_class,
        "error":       (error or "")[:500],
        "timestamp":   time.time(),
    }
    if extra:
        entry.update(extra)
    try:
        await r.xadd(DLQ_STREAM, entry, maxlen=5000, approximate=True)
    except Exception as dlq_err:
        logger.warning(
            "dlq_write_failed",
            extra={"event": "dlq_write_failed", "message_id": message_id,
                   "error": str(dlq_err)},
        )


async def _timeout_sweeper(r) -> None:
    """
    Periodically scan pending-aggregation keys and DLQ any message whose 3 ML
    partials never all arrived within AGGREGATION_TIMEOUT_MS — previously these
    were lost silently when the 30s Redis TTL expired.
    """
    timeout_s = AGGREGATION_TIMEOUT_MS / 1000.0
    while True:
        try:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            now = time.time()
            async for key in r.scan_iter(match=f"{PENDING_KEY_PREFIX}*", count=100):
                fields = await r.hgetall(key)
                if not fields:
                    continue
                arrival = float(fields.get("arrival_timestamp", now))
                if now - arrival < timeout_s:
                    continue  # still within the collection window

                received = [k for k in fields.keys() if k != "arrival_timestamp"]
                if all(m in received for m in EXPECTED_MODELS):
                    continue  # complete but not yet cleaned up — leave for the loop

                msg_id  = key.split(PENDING_KEY_PREFIX, 1)[-1]
                missing = list(set(EXPECTED_MODELS) - set(received))
                logger.warning(
                    "aggregation_timeout",
                    extra={
                        "event":      "aggregation_timeout",
                        "message_id": msg_id,
                        "missing":    missing,
                        "received":   received,
                        "age_ms":     round((now - arrival) * 1000, 1),
                    },
                )
                await _write_dlq(
                    r, msg_id, reason="aggregation_timeout",
                    error_class="TimeoutError",
                    error=f"missing models {missing} after {round((now-arrival)*1000)}ms",
                    extra={"missing": json.dumps(missing)},
                )
                await r.delete(key)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.log_exception("AGGREGATION TIMEOUT SWEEPER ERROR", e)


async def run_consumer_loop(redis_client: RedisClient, get_meta_learner,
                            trajectory_model) -> None:
    """
    Main event loop — blocks on xreadgroup and dispatches aggregation when all 3
    model partials for a message have arrived.

    get_meta_learner: callable returning the current META_LEARNER (supports
                      hot-reload without restarting the loop).
    trajectory_model: the loaded ConversationLSTM (or None).
    """
    r = redis_client.redis

    try:
        await r.xgroup_create(INPUT_STREAM, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating consumer group: {e}")

    sweeper = asyncio.create_task(_timeout_sweeper(r))
    logger.info("Central Responder consumer loop started.")

    try:
        while True:
            try:
                streams  = {INPUT_STREAM: ">"}
                messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams,
                                              count=10, block=2000)

                if messages:
                    for stream, msgs in messages:
                        for record_id, data in msgs:
                            await _handle_record(r, redis_client, record_id, data,
                                                 get_meta_learner, trajectory_model)

            except Exception as e:
                logger.log_exception("CENTRAL RESPONDER CRITICAL ERROR", e)
                await asyncio.sleep(1)
    finally:
        sweeper.cancel()


async def _handle_record(r, redis_client, record_id, data,
                         get_meta_learner, trajectory_model) -> None:
    """Process one partial-analysis stream record."""
    msg_id     = data.get("message_id")
    model_name = data.get("model_name")
    rlog       = logger.bind(message_id=msg_id, model=model_name)

    scores_raw = data.get("scores")
    scores = scores_raw
    if isinstance(scores_raw, str):
        try:
            scores = json.loads(scores_raw)
        except (json.JSONDecodeError, TypeError) as e:
            rlog.warning("scores_parse_failed",
                         extra={"event": "scores_parse_failed",
                                "error_class": type(e).__name__})

    full_packet = {
        "model_name":    model_name,
        "scores":        scores,
        "original_data": data.get("original_data"),
        "emoji_scores":  data.get("emoji_scores", "{}"),
    }
    if isinstance(full_packet["original_data"], str):
        try:
            full_packet["original_data"] = json.loads(full_packet["original_data"])
        except (json.JSONDecodeError, TypeError) as e:
            rlog.warning("original_data_parse_failed",
                         extra={"event": "original_data_parse_failed",
                                "error_class": type(e).__name__})

    pending_key = f"{PENDING_KEY_PREFIX}{msg_id}"

    is_new = not await r.exists(pending_key)
    if is_new:
        await r.hset(pending_key, "arrival_timestamp", time.time())
    await r.hset(pending_key, model_name, json.dumps(full_packet))
    await r.expire(pending_key, 30)  # refresh TTL on every update

    current_results = await r.hgetall(pending_key)
    received_models = [k for k in current_results.keys() if k != "arrival_timestamp"]

    if all(m in received_models for m in EXPECTED_MODELS):
        rlog.debug("all_models_received", extra={"event": "all_models_received"})

        arrival_ts = float(current_results.get("arrival_timestamp", time.time()))
        agg_lat    = (time.time() - arrival_ts) * 1000

        all_packets = []
        for m, packet_str in current_results.items():
            if m == "arrival_timestamp":
                continue
            try:
                all_packets.append(json.loads(packet_str))
            except Exception as e:
                rlog.warning("packet_parse_failed",
                             extra={"event": "packet_parse_failed", "packet_model": m,
                                    "error_class": type(e).__name__})

        try:
            await aggregate_and_publish(
                msg_id, all_packets, get_meta_learner(), trajectory_model,
                redis_client, r, agg_lat=agg_lat,
            )
        except Exception as agg_err:
            # Don't lose the message: log with traceback (bound) AND DLQ it. The
            # xack still fires below so the consumer group doesn't redeliver forever.
            rlog.log_exception("AGGREGATION FAILED", agg_err)
            await _write_dlq(r, msg_id, reason="aggregation_error",
                             error_class=type(agg_err).__name__, error=str(agg_err))
    else:
        missing = list(set(EXPECTED_MODELS) - set(received_models))
        rlog.debug("waiting_for_models",
                   extra={"event": "waiting_for_models", "missing": missing})

    # ACK the stream record regardless of outcome (failures are captured in the DLQ).
    await r.xack(INPUT_STREAM, GROUP_NAME, record_id)
