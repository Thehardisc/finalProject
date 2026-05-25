import asyncio
import sys
import os
import json
import time

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = get_logger("vader_service")

class VaderAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)
        return {
            "vader_neg": scores["neg"],
            "vader_neu": scores["neu"],
            "vader_pos": scores["pos"],
            "vader_compound": scores["compound"]
        }

redis_client = RedisClient()
STREAM_KEY = "preprocessed_stream"
GROUP_NAME = "vader_group"
CONSUMER_NAME = "vader_worker_1"
OUTPUT_STREAM = "partial_analysis_stream"
MODEL_NAME = "vader"

analyzer = VaderAnalyzer()

async def main():
    await redis_client.connect()
    r = redis_client.redis
    
    # Create consumer group if not exists
    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    logger.info(f"{MODEL_NAME} Analysis worker started.")
    
    while True:
        try:
            streams = {STREAM_KEY: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=1, block=2000)

            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        try:
                            text_to_analyze = data.get("text", "") or data.get("original_text", "")
                            start_time = time.time()
                            logger.debug(f"Analyzing message {message_id} with {MODEL_NAME}...")

                            scores = analyzer.analyze(text_to_analyze)
                            elapsed = (time.time() - start_time) * 1000

                            stats = {
                                "Message ID": data.get("message_id", "N/A"),
                                "Text Snippet": (text_to_analyze[:30] + '...') if len(text_to_analyze) > 30 else text_to_analyze,
                                "Latency": f"{elapsed:.2f}ms",
                                "VADER Compound": scores["vader_compound"],
                                "Positive": scores["vader_pos"],
                                "Negative": scores["vader_neg"]
                            }
                            logger.log_stats(f"{MODEL_NAME.upper()} Inference", stats)

                            output_event = {
                                "message_id": data.get("message_id", message_id),
                                "original_data": data,
                                "model_name": MODEL_NAME,
                                "scores": scores,
                                "latency_ms": elapsed
                            }
                            await redis_client.publish_event(OUTPUT_STREAM, output_event)
                            await r.xack(STREAM_KEY, GROUP_NAME, message_id)

                        except Exception as msg_err:
                            logger.error(f"[VADER] Failed on {message_id}: {msg_err}. ACKing to prevent requeue.")
                            try:
                                await r.xack(STREAM_KEY, GROUP_NAME, message_id)
                            except Exception:
                                pass

        except Exception as e:
            if "NOGROUP" in str(e):
                try:
                    await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
                    logger.info("Re-created consumer group after NOGROUP error.")
                except Exception as cg_err:
                    if "BUSYGROUP" not in str(cg_err):
                        logger.error(f"Failed to re-create group: {cg_err}")
            else:
                logger.log_exception("VADER WORKER — Redis error, retrying in 1s", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
