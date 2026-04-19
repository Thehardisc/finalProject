import asyncio
import sys
import os
import json

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from preprocessing_service.utils import clean_text, demojize_text

logger = get_logger("preprocessing_service")

redis_client = RedisClient()
STREAM_KEY = "message_stream"
GROUP_NAME = "preprocessing_group"
CONSUMER_NAME = "worker_1"
OUTPUT_STREAM = "preprocessed_stream"

async def main():
    await redis_client.connect()
    r = redis_client.redis
    
    # Create consumer group if not exists
    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    logger.info("Preprocessing worker started.")
    
    while True:
        try:
            # Read new messages
            streams = {STREAM_KEY: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=1, block=2000)
            
            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        # Process message
                        logger.debug(f"Processing message {message_id}")
                        
                        original_text = data.get("text", "")
                        
                        # Generate Dual Payloads
                        # 1. Raw (Cleaned + Emojis) -> For VADER
                        processed_text = clean_text(original_text)
                        
                        # 2. Demojized (Cleaned + Text Emojis) -> For BERT/Sentiment
                        processed_text_demojized = demojize_text(processed_text)
                        
                        # Prepare output event
                        output_event = data.copy()
                        output_event["processed_text"] = processed_text
                        output_event["processed_text_demojized"] = processed_text_demojized
                        output_event["original_text"] = original_text # Ensure we keep original
                        
                        # Publish to next stage
                        await redis_client.publish_event(OUTPUT_STREAM, output_event)
                        
                        # Structured Stats Reporting
                        stats = {
                            "Message ID": data.get("message_id", "N/A"),
                            "Original": (original_text[:40] + '...') if len(original_text) > 40 else original_text,
                            "Demojized": (processed_text_demojized[:40] + '...') if len(processed_text_demojized) > 40 else processed_text_demojized
                        }
                        logger.log_stats("PREPROCESSING COMPLETE", stats)
                        
                        # Ack message
                        await r.xack(STREAM_KEY, GROUP_NAME, message_id)
            
        except Exception as e:
            logger.log_exception("PREPROCESSING WORKER FATAL ERROR", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
