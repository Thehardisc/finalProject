import asyncio
import sys
import os
import json
import time
import torch
from collections import Counter
from transformers import pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import emoji as emoji_lib
from shared.constants import EMOTION_LABELS
from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger

logger = get_logger("emojinet_service")

redis_client = RedisClient()
INPUT_STREAM  = "preprocessed_stream"
GROUP_NAME    = "emojinet_group"
CONSUMER_NAME = "emojinet_worker_1"
OUTPUT_STREAM = "partial_analysis_stream"
MODEL_NAME    = "emojinet"


class EmojiNetAnalyzer:
    def __init__(self):
        logger.info("Loading GoEmotions model for EmojiNet...")
        device = -1
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"

        self.classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-go-emotion",
            top_k=None,
            device=device,
        )
        logger.info("EmojiNet model loaded.")

    def analyze(self, text: str) -> dict:
        occurrences = emoji_lib.emoji_list(text)
        if not occurrences:
            return {label: 0.0 for label in EMOTION_LABELS}

        counts = Counter(e["emoji"] for e in occurrences)
        total  = sum(counts.values())
        weighted = {label: 0.0 for label in EMOTION_LABELS}

        for ch, cnt in counts.items():
            desc = emoji_lib.demojize(ch).strip(":").replace("_", " ")
            if not desc:
                continue
            try:
                result = {r["label"]: r["score"] for r in self.classifier(desc[:512])[0]}
            except Exception as e:
                logger.warning(f"Model error for emoji '{ch}': {e}")
                continue
            w = cnt / total
            for label in EMOTION_LABELS:
                weighted[label] += result.get(label, 0.0) * w

        return weighted


logger.info("Initializing EmojiNet analyzer...")
analyzer = EmojiNetAnalyzer()


async def main():
    await redis_client.connect()
    r = redis_client.redis

    try:
        await r.xgroup_create(INPUT_STREAM, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    logger.log_stats("EMOJINET Service Status", {
        "Model":  "bhadresh-savani/bert-base-go-emotion (28 classes)",
        "Device": str(analyzer.classifier.device),
        "Status": "READY",
        "Stream": INPUT_STREAM,
    })
    logger.info("EmojiNet analysis worker started.")

    while True:
        try:
            streams  = {INPUT_STREAM: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=1, block=2000)

            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        try:
                            text     = data.get("text", "") or data.get("original_text", "")
                            id_val   = data.get("message_id", message_id)
                            start_ts = time.time()

                            scores  = analyzer.analyze(text)
                            elapsed = (time.time() - start_ts) * 1000

                            found = emoji_lib.emoji_list(text)
                            if found:
                                top = max(scores.items(), key=lambda x: x[1])
                                logger.log_stats("EMOJINET Inference", {
                                    "Message ID":   id_val,
                                    "Emojis":       " ".join(e["emoji"] for e in found),
                                    "Top Emotion":  f"{top[0]} ({top[1]:.2%})",
                                    "Latency":      f"{elapsed:.2f}ms",
                                })
                            else:
                                logger.debug(f"EmojiNet: no emojis in {id_val}, publishing zeros.")

                            output_event = {
                                "message_id":   id_val,
                                "original_data": json.dumps(dict(data)),
                                "model_name":   MODEL_NAME,
                                "scores":       json.dumps(scores),
                                "latency_ms":   elapsed,
                            }
                            await redis_client.publish_event(OUTPUT_STREAM, output_event)
                            await r.xack(INPUT_STREAM, GROUP_NAME, message_id)

                        except Exception as msg_err:
                            logger.error(f"[EMOJINET] Failed on {message_id}: {msg_err}. ACKing.")
                            try:
                                await r.xack(INPUT_STREAM, GROUP_NAME, message_id)
                            except Exception:
                                pass

        except Exception as e:
            logger.log_exception("EMOJINET WORKER — Redis error, retrying in 1s", e)
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
