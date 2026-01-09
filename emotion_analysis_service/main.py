import asyncio
import sys
import os
import json

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from emotion_analysis_service.analyzers.vader_analyzer import VaderAnalyzer
from emotion_analysis_service.analyzers.go_emotions_analyzer import GoEmotionsAnalyzer
from emotion_analysis_service.analyzers.sentiment_analyzer import SentimentAnalyzer
from emotion_analysis_service.analyzers.rule_analyzer import RuleBasedAnalyzer

redis_client = RedisClient()
STREAM_KEY = "preprocessed_stream"
GROUP_NAME = "analysis_group"
CONSUMER_NAME = "worker_1"
OUTPUT_STREAM = "emotion_stream"

vader = VaderAnalyzer()
bert = GoEmotionsAnalyzer()
sentiment = SentimentAnalyzer()
rules = RuleBasedAnalyzer()

async def main():
    await redis_client.connect()
    r = redis_client.redis
    
    # Create consumer group if not exists
    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            print(f"Error creating group: {e}")

    print("Emotion Analysis worker started...")
    
    while True:
        try:
            # Read new messages
            streams = {STREAM_KEY: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=1, block=2000)
            
            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        # Process message
                        text_to_analyze = data.get("processed_text", "")
                        original_text = data.get("original_text", "") # We need this for CAPS detection
                        print(f"Analyzing message {message_id}: {text_to_analyze}")
                        
                        # Run Analyzers
                        vader_scores = vader.analyze(text_to_analyze)
                        bert_scores = bert.analyze(text_to_analyze)
                        sentiment_scores = sentiment.analyze(text_to_analyze)
                        rule_scores = rules.analyze(text_to_analyze, original_text=original_text)
                        
                        # Merge Results
                        combined_emotions = {**vader_scores, **bert_scores, **sentiment_scores, **rule_scores}
                        
                        # Prepare output event
                        output_event = data.copy()
                        output_event["emotions"] = json.dumps(combined_emotions) # storing as json string or nested obj if streams allow? Redis streams store strings.
                        
                        # Publish to next stage
                        await redis_client.publish_event(OUTPUT_STREAM, output_event)
                        
                        # Ack message
                        await r.xack(STREAM_KEY, GROUP_NAME, message_id)
            
        except Exception as e:
            print(f"Error in processing loop: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
