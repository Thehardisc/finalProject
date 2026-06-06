"""
shared/nlp_worker.py — Generic Redis consumer loop for NLP analyzer services.

Usage:
    asyncio.run(run_nlp_worker(
        model_name   = "vader",
        group_name   = "vader_group",
        consumer_name= "vader_worker_1",
        analyzer     = MyAnalyzer(),        # must have .analyze(text: str) -> dict
        get_text     = lambda d: d.get("text", ""),
        format_stats = lambda d, scores, ms: {...},  # optional
        logger       = get_logger("vader_service"),  # optional
    ))
"""

import asyncio
import time

from .utils.redis_client import RedisClient
from .utils.logger import get_logger
from .module_registry import register_module

STREAM_KEY = "preprocessed_stream"
OUTPUT_STREAM = "partial_analysis_stream"


async def run_nlp_worker(
    model_name: str,
    group_name: str,
    consumer_name: str,
    analyzer,
    get_text,
    format_stats=None,
    logger=None,
):
    if logger is None:
        logger = get_logger(model_name)

    redis_client = RedisClient()
    await redis_client.connect()
    r = redis_client.redis

    try:
        await r.xgroup_create(STREAM_KEY, group_name, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating consumer group: {e}")

    await register_module(r, model_name, required=True, schema="scores", logger=logger)
    logger.info(f"{model_name} analysis worker started.")

    while True:
        try:
            messages = await r.xreadgroup(
                group_name, consumer_name, {STREAM_KEY: ">"}, count=1, block=2000
            )

            if messages:
                for _stream, msgs in messages:
                    for message_id, data in msgs:
                        try:
                            text = get_text(data)
                            start = time.time()
                            logger.debug(f"Analyzing message {message_id} with {model_name}...")

                            scores = analyzer.analyze(text)
                            elapsed = (time.time() - start) * 1000

                            if format_stats:
                                stats = format_stats(data, scores, elapsed)
                                logger.log_stats(f"{model_name.upper()} Inference", stats)

                            await redis_client.publish_event(OUTPUT_STREAM, {
                                "message_id": data.get("message_id", message_id),
                                "original_data": data,
                                "model_name": model_name,
                                "scores": scores,
                                "latency_ms": elapsed,
                            })
                            await r.xack(STREAM_KEY, group_name, message_id)

                        except Exception as msg_err:
                            logger.error(
                                f"[{model_name.upper()}] Failed on {message_id}: {msg_err}. ACKing to prevent requeue.",
                                exc_info=True,
                            )
                            try:
                                await r.xack(STREAM_KEY, group_name, message_id)
                            except Exception:
                                pass

        except Exception as e:
            if "NOGROUP" in str(e):
                try:
                    await r.xgroup_create(STREAM_KEY, group_name, mkstream=True)
                    logger.info("Re-created consumer group after NOGROUP error.")
                except Exception as cg_err:
                    if "BUSYGROUP" not in str(cg_err):
                        logger.error(f"Failed to re-create consumer group: {cg_err}")
            else:
                logger.log_exception(f"{model_name.upper()} WORKER — Redis error, retrying in 1s", e)
            await asyncio.sleep(1)
