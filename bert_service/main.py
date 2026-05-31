import asyncio
import sys
import os
import json
import time

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from shared.module_registry import register_module
from transformers import pipeline

logger = get_logger("bert_service")

class BasicBertAnalyzer:
    def __init__(self):
        import torch
        # Determine device: CUDA (Nvidia), MPS (Mac), or CPU
        device = -1
        if torch.cuda.is_available():
            device = 0
            logger.info("BERT using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("BERT using Apple Metal (MPS) GPU")
        else:
            logger.info("BERT using CPU")
        # Using a small, fast model for basic emotions (Ekman)
        # "j-hartmann/emotion-english-distilroberta-base" covers:
        # anger, disgust, fear, joy, neutral, sadness, surprise
        self.classifier = pipeline("text-classification", 
                                   model="j-hartmann/emotion-english-distilroberta-base", 
                                   return_all_scores=True,
                                   device=device)
    
    def analyze(self, text: str) -> dict:
        # Truncate to 512 chars to prevent transformer sequence length errors
        if len(text) > 512:
            text = text[:512]
        results = self.classifier(text)
        # results is [[{'label': 'anger', 'score': 0.9}, ...]]
        scores = {}
        for r in results[0]:
            scores[r['label']] = r['score']
        return scores

redis_client = RedisClient()
STREAM_KEY = "preprocessed_stream"
GROUP_NAME = "bert_group"
CONSUMER_NAME = "bert_worker_1"
OUTPUT_STREAM = "partial_analysis_stream"
MODEL_NAME = "basic_bert"

# Initialize model outside loop
logger.info("Loading BERT model...")
analyzer = BasicBertAnalyzer()
logger.info("BERT model loaded.")

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
        "Model": "DistilRoBERTa-Base (Ekman)",
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
                        mid  = data.get("message_id", "")
                        cid  = data.get("conversation_id", "")
                        uid  = data.get("user_id", "")
                        mlog = logger.bind(message_id=mid, conversation_id=cid, user_id=uid, model=MODEL_NAME)
                        try:
                            text_to_analyze = data.get("processed_text_demojized", "") or data.get("processed_text", "")
                            start_time = time.time()
                            mlog.debug("bert_start", extra={"event": "model_start"})

                            scores = analyzer.analyze(text_to_analyze)
                            elapsed = (time.time() - start_time) * 1000

                            top_emo = max(scores.items(), key=lambda x: x[1])
                            mlog.info(
                                "bert_done",
                                extra={
                                    "event":        "model_done",
                                    "latency_ms":   round(elapsed, 2),
                                    "dominant":     top_emo[0],
                                    "confidence":   round(top_emo[1], 4),
                                    "text_len":     len(text_to_analyze),
                                },
                            )

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
                            mlog.error(
                                "bert_failed",
                                extra={
                                    "event":       "model_failed",
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
                    logger.info("Re-created consumer group after NOGROUP error.")
                except Exception as cg_err:
                    if "BUSYGROUP" not in str(cg_err):
                        logger.error(f"Failed to re-create group: {cg_err}")
            else:
                logger.log_exception(f"{MODEL_NAME.upper()} WORKER — Redis error, retrying in 1s", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
