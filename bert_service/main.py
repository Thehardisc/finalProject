import asyncio
import sys
import os
import json

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from transformers import pipeline

class BasicBertAnalyzer:
    def __init__(self):
        # Using a small, fast model for basic emotions (Ekman)
        # "j-hartmann/emotion-english-distilroberta-base" covers:
        # anger, disgust, fear, joy, neutral, sadness, surprise
        self.classifier = pipeline("text-classification", 
                                   model="j-hartmann/emotion-english-distilroberta-base", 
                                   return_all_scores=True)
    
    def analyze(self, text: str) -> dict:
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
print("Loading BERT model...")
analyzer = BasicBertAnalyzer()
print("BERT model loaded.")

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
                        # BERT models prefer demojized text
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
