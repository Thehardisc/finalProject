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

# Load Emoji Map
EMOJI_MAP_PATH = os.path.join(os.path.dirname(__file__), 'analyzers', 'emoji_map.json')
emoji_boost_map = {}
try:
    with open(EMOJI_MAP_PATH, 'r') as f:
        emoji_boost_map = json.load(f).get("keywords", {})
    print(f"Loaded {len(emoji_boost_map)} emoji boosting rules.")
except Exception as e:
    print(f"Warning: Could not load emoji map: {e}")

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
                        text_raw = data.get("processed_text", "")
                        text_demojized = data.get("processed_text_demojized", text_raw) # Fallback to raw if missing
                        original_text = data.get("original_text", "")

                        print(f"Analyzing message {message_id}...")
                        
                        # Run Analyzers
                        # VADER handles raw emojis well
                        vader_scores = vader.analyze(text_raw)
                        
                        # BERT models (GoEmotions, Sentiment) prefer text-aliases
                        # BERT models (GoEmotions, Sentiment) prefer text-aliases
                        bert_scores = bert.analyze(text_demojized)
                        sentiment_scores = sentiment.analyze(text_demojized)
                        
                        # --- CONTEXTUAL CONFLICT RESOLUTION (The "Fusion" Layer) ---
                        # General logic to resolve conflicts between Text Sentiment and Emoji Sentiment
                        # Handles Sarcasm (Neg Text + Pos Emoji) and Slang (Pos Text + Neg Emoji)
                        
                        POSITIVE_EMOTIONS = {'joy', 'love', 'admiration', 'optimism', 'amusement', 'excitement', 'approval', 'gratitude', 'relief', 'desire'}
                        NEGATIVE_EMOTIONS = {'anger', 'annoyance', 'disapproval', 'disgust', 'sadness', 'remorse', 'fear', 'nervousness', 'grief'}

                        def resolve_context_conflict(bert_s, vader_s, text_aliased):
                            output = bert_s.copy()
                            text_lower = text_aliased.lower()
                            reasoning = None
                            
                            # 1. Calculate Vectors
                            # Text Vector (Vt)
                            vt = vader_s.get('vader_compound', 0)
                            
                            # Emoji Vector (Ve) - Heuristic based on map
                            ve_pos_score = 0
                            ve_neg_score = 0
                            
                            # First apply standard boosting (Signal Boosting)
                            # We still do this to get the base "literal" signal from emojis
                            for keyword, boosts in emoji_boost_map.items():
                                if keyword in text_lower:
                                    for emotion, increment in boosts.items():
                                        if emotion in output:
                                            output[emotion] = min(1.0, output[emotion] + increment)
                                        
                                        # Accumulate Emoji Polarity for Context Check
                                        if emotion in POSITIVE_EMOTIONS:
                                            ve_pos_score += increment
                                        elif emotion in NEGATIVE_EMOTIONS:
                                            ve_neg_score += increment
                            
                            ve = ve_pos_score - ve_neg_score # Net Emoji Sentiment

                            # 2. General Conflict Resolution Logic
                            
                            # CASE A: SARCASM DETECTED
                            # Condition: Negative Text (Vt < -0.3) + Positive Emoji (Ve > 0.3)
                            # Example: "I hate this 😂"
                            if vt < -0.3 and ve > 0.3:
                                print(f"  [Context] Sarcasm Detected! (Text: {vt}, Emoji: {ve})")
                                reasoning = {
                                    "type": "Sarcasm Detected",
                                    "details": f"Negative Text ({vt:.2f}) + Positive Emoji ({ve:.2f})",
                                    "action": "Inverted negative emotions, boosted Amusement."
                                }
                                # Action: Invert Negative Emotions, Boost Amusement
                                for emo in NEGATIVE_EMOTIONS:
                                    if emo in output: output[emo] *= 0.2 # Suppress significantly
                                
                                if 'amusement' in output: output['amusement'] = min(1.0, output['amusement'] + 0.5)
                                if 'joy' in output: output['joy'] = min(1.0, output['joy'] + 0.3)

                            # CASE B: SLANG / INTENSITY
                            # Condition: Positive Text (Vt > 0.3) + Negative Emoji (Ve < -0.3)
                            # Example: "I love you ☠️" (Skull maps to Fear/Death literally)
                            elif vt > 0.3 and ve < -0.3:
                                print(f"  [Context] Slang/Intensity Detected! (Text: {vt}, Emoji: {ve})")
                                reasoning = {
                                    "type": "Slang/Intensity",
                                    "details": f"Positive Text ({vt:.2f}) + Negative Emoji ({ve:.2f})",
                                    "action": "Interpreted negative emoji as Intensity/Excitement."
                                }
                                # Action: Suppress Literal Fear/Negativity, Boost Excitement/Amusement
                                for emo in NEGATIVE_EMOTIONS:
                                    if emo in output: output[emo] *= 0.1 # Suppress hard (it's not real fear)
                                
                                # Reinterpret the intensity as Amusement or Excitement
                                if 'amusement' in output: output['amusement'] = min(1.0, output['amusement'] + 0.6)
                                if 'excitement' in output: output['excitement'] = min(1.0, output['excitement'] + 0.5)
                                
                            return output, reasoning

                        bert_scores, resolution_reasoning = resolve_context_conflict(bert_scores, vader_scores, text_demojized)
                        # -----------------------
                        sentiment_scores = sentiment.analyze(text_demojized)
                        
                        rule_scores = rules.analyze(text_raw, original_text=original_text)
                        
                        # Merge Results
                        combined_emotions = {**bert_scores, **sentiment_scores, **rule_scores} # We used updated bert_scores
                        
                        # Prepare output event
                        output_event = data.copy()
                        output_event["emotions"] = json.dumps(combined_emotions)
                        if resolution_reasoning:
                            output_event["reasoning"] = json.dumps(resolution_reasoning)
                        
                        # Publish to next stage
                        await redis_client.publish_event(OUTPUT_STREAM, output_event)
                        
                        # Ack message
                        await r.xack(STREAM_KEY, GROUP_NAME, message_id)
            
        except Exception as e:
            print(f"Error in processing loop: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
