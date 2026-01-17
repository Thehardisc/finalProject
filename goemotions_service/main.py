import asyncio
import sys
import os
import json
import torch
from transformers import pipeline

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient

class GoEmotionsAnalyzer:
    def __init__(self):
        print("Loading GoEmotions BERT model...")
        
        # Determine device: CUDA (Nvidia), MPS (Mac), or CPU
        device = -1
        if torch.cuda.is_available():
            device = 0 # CUDA device 0
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Metal (MPS) GPU")
        else:
            print("Using CPU")

        # Using a model fine-tuned on GoEmotions (28 labels)
        self.classifier = pipeline(
            "text-classification", 
            model="bhadresh-savani/bert-base-go-emotion", 
            top_k=None, 
            device=device
        )
        print("GoEmotions BERT model loaded.")
    
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

print("Initializing GoEmotions...")
analyzer = GoEmotionsAnalyzer()

async def main():
    await redis_client.connect()
    r = redis_client.redis
    
    # Create consumer group if not exists
    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            print(f"Error creating group: {e}")

    print(f"{MODEL_NAME} Analysis worker started...")
    
    while True:
        try:
            # Read new messages
            streams = {STREAM_KEY: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=1, block=2000)
            
            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        # Process message
                        text_to_analyze = data.get("processed_text_demojized", "") or data.get("processed_text", "")
                        
                        print(f"Analyzing message {message_id} with {MODEL_NAME}...")
                        
                        scores = analyzer.analyze(text_to_analyze)
                        
                        output_event = {
                            "message_id": data.get("message_id", message_id), 
                            "original_data": data, 
                            "model_name": MODEL_NAME,
                            "scores": scores
                        }
                        
                        await redis_client.publish_event(OUTPUT_STREAM, output_event)
                        
                        # Ack message
                        await r.xack(STREAM_KEY, GROUP_NAME, message_id)
            
        except Exception as e:
            print(f"Error in processing loop: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
