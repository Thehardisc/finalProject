import asyncio
import sys
import os
import json
import time
import torch
from transformers import pipeline

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from shared.module_registry import register_module

logger = get_logger("goemotions_service")

class GoEmotionsAnalyzer:
    def __init__(self):
        logger.info("Loading GoEmotions BERT model...")
        
        # Determine device: CUDA (Nvidia), MPS (Mac), or CPU
        device = -1
        if torch.cuda.is_available():
            device = 0 # CUDA device 0
            logger.info("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal (MPS) GPU")
        else:
            logger.info("Using CPU")

        # Using a model fine-tuned on GoEmotions (28 labels)
        self.classifier = pipeline(
            "text-classification", 
            model="bhadresh-savani/bert-base-go-emotion", 
            top_k=None, 
            device=device
        )
        logger.info("GoEmotions BERT model loaded.")
    
    def analyze(self, text: str) -> dict:
        if len(text) > 512:
            text = text[:512]
            
        results = self.classifier(text)[0]
        emotions = {res['label']: res['score'] for res in results}
        
        return emotions

redis_client = RedisClient()
STREAM_KEY = "preprocessed_stream"
GROUP_NAME = "goemotions_group"
CONSUMER_NAME = "goemotions_worker_1"
OUTPUT_STREAM = "partial_analysis_stream"
MODEL_NAME = "go_emotions"

logger.info("Initializing GoEmotions...")
analyzer = GoEmotionsAnalyzer()

async def main():
    await redis_client.connect()
    r = redis_client.redis
    
    # Create consumer group if not exists
    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    # Log Ready State
    logger.log_stats(f"{MODEL_NAME.upper()} Service Status", {
        "Model": "BERT-Base (bhadresh-savani/bert-base-go-emotion, 28 classes)",
        "Device": analyzer.classifier.device.type,
        "Status": "READY",
        "Stream": STREAM_KEY
    })
    await register_module(r, MODEL_NAME, required=True, schema="scores", logger=logger)
    logger.info(f"{MODEL_NAME} Analysis worker started.")

    while True:
        try:
            streams = {STREAM_KEY: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=1, block=2000)

            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        try:
                            text_to_analyze = data.get("processed_text_demojized", "") or data.get("processed_text", "")
                            start_time = time.time()
                            logger.debug(f"Analyzing message {message_id} with {MODEL_NAME}...")

                            scores = analyzer.analyze(text_to_analyze)
                            elapsed = (time.time() - start_time) * 1000

                            top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                            stats = {
                                "Message ID": data.get("message_id", "N/A"),
                                "Top Emotions": ", ".join([f"{e} ({s:.2%})" for e, s in top_3]),
                                "Inference Time": f"{elapsed:.2f}ms"
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
                            logger.error(f"[GOEMOTIONS] Failed on {message_id}: {msg_err}. ACKing to prevent requeue.", exc_info=True)
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
                logger.log_exception(f"{MODEL_NAME.upper()} WORKER — Redis error, retrying in 1s", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
