import asyncio
import sys
import os
import json
import time
from datetime import datetime

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger

logger = get_logger("aggregation_service")

redis_client = RedisClient()
STREAM_KEY = "emotion_stream"
GROUP_NAME = "aggregation_group"
CONSUMER_NAME = "worker_1"
OUTPUT_STREAM = "conversation_update_stream"

import re

async def handle_dynamic_rules(conversation_id: str, text: str, r) -> tuple:
    """
    1. Parses text for new rules (e.g. "when i say X it means Y").
    2. Stores rules in Redis.
    3. Checks text against existing rules and returns an override emotion if matched.
    """
    rule_key = f"conversation:{conversation_id}:rules"
    
    # 1. Parse for NEW rules
    # Patterns:
    # "when i say <trigger> it means <meaning>"
    # "<trigger> means <meaning>"
    # "define <trigger> as <meaning>"
    patterns = [
        r"when i say (.+) it means (.+)",
        r"(.+) means (.+)",
        r"define (.+) as (.+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            trigger = match.group(1).strip().lower()
            meaning = match.group(2).strip().lower()
            
            logger.info(f"LEARNING RULE: '{trigger}' -> '{meaning}'")
            await r.hset(rule_key, trigger, meaning)
            return None, None 

    # 2. Check for EXISTING rules
    rules = await r.hgetall(rule_key) # returns {trigger: meaning}
    logger.debug(f"Checking {len(rules)} rules for text: '{text}'")
    
    if rules:
        text_lower = text.lower()
        for trigger, meaning in rules.items():
            if trigger in text_lower:
                logger.info(f"RULE MATCHED: '{trigger}' detected. Overriding with meaning '{meaning}'")
                return trigger, meaning
                
    return None, None

async def update_conversation_state(conversation_id: str, new_emotions: dict, r, original_text: str = ""):
    # Retrieve current state
    state_key = f"conversation:{conversation_id}"
    current_state = await r.hgetall(state_key)
    
    if not current_state:
        # Initialize state
        current_state = {
            "message_count": 0,
            "accumulated_valence": 0.0,
            "accumulated_arousal": 0.0, # Placeholder, VADER doesn't give arousal directly usually, but we can treat compound as valence
            "last_updated": 0
        }
    
    # Parse values
    msg_count = int(current_state.get("message_count", 0))
    acc_valence = float(current_state.get("accumulated_valence", 0.0))
    prev_mood = current_state.get("overall_mood", "Neutral")
    
    # Check for Dynamic Rules
    override_trigger, override_meaning = await handle_dynamic_rules(conversation_id, original_text, r)
    
    # Update logic (Simple averaging for this MVP)
    # Using VADER compound score as valence
    new_valence = new_emotions.get("vader_compound", 0.0)
    
    # Use dominant emotion from Central Responder if available
    dominant_emotion = new_emotions.get("dominant_emotion")
    dominant_score = 1.0 # If we trust the central responder, we can treat it as high confidence, or extract score from map
    
    if not dominant_emotion:
        # Fallback to old logic if field missing
        latest_emotions = {k: v for k, v in new_emotions.items() if k not in ["vader_neg", "vader_neu", "vader_pos", "vader_compound", "dominant_emotion"]}
        dominant_emotion = "None"
        dominant_score = 0.0
        for emo, score in latest_emotions.items():
            if score > dominant_score:
                dominant_score = score
                dominant_emotion = emo
            
    # APPLY OVERRIDE
    if override_meaning:
        logger.info(f"Applying override logic for meaning: {override_meaning}")
        # Heuristic: If meaning contains positive words, force positive valence/emotion
        # This is a simple interpretation layer
        meaning_lower = override_meaning.lower()
        if any(w in meaning_lower for w in ["love", "happy", "good", "great", "joy"]):
            new_valence = 0.9
            dominant_emotion = "love" # Force neural emotion
            dominant_score = 1.0
        elif any(w in meaning_lower for w in ["hate", "bad", "angry", "kill", "sad"]):
            new_valence = -0.9
            dominant_emotion = "anger"
            dominant_score = 1.0
        else:
             # Just set it to the meaning text itself as a custom emotion?
             # For now, let's keep it simple.
             pass
    
    msg_count += 1
    acc_valence += new_valence
    avg_valence = acc_valence / msg_count
    
    # Determine dominant emotion state with finer granularity
    if avg_valence >= 0.6:
        overall_mood = "Ecstatic"
    elif avg_valence >= 0.2:
        overall_mood = "Positive"
    elif avg_valence >= 0.05:
        overall_mood = "Slightly Positive"
    elif avg_valence > -0.05:
        overall_mood = "Neutral"
    elif avg_valence > -0.2:
        overall_mood = "Slightly Negative"
    elif avg_valence > -0.6:
        overall_mood = "Negative"
    else:
        overall_mood = "Hostile"
 
    if overall_mood != prev_mood:
        logger.info(f"[State Shift] Conversation {conversation_id}: MOOD TRANSITION | {prev_mood} -> {overall_mood}")
        logger.info(f"             Reasoning: Current Average Valence is {avg_valence:.4f}")

    # Save new state
    new_state = {
        "message_count": msg_count,
        "accumulated_valence": acc_valence,
        "average_valence": avg_valence,
        "overall_mood": overall_mood,
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_score": dominant_score,
        "last_updated": time.time(),
        "last_message_emotions": json.dumps(new_emotions)
    }
    
    # Store in Redis
    # Redis hset expects string values for mapping
    # We convert numbers to strings implicitly or explicitly
    mapping_to_store = {k: str(v) for k, v in new_state.items()}
    await r.hset(state_key, mapping=mapping_to_store)
 
    # Log Aggregation Stats
    stats = {
        "Session ID": conversation_id,
        "Total Messages": msg_count,
        "Last Valence": f"{new_valence:+.4f}",
        "Average Valence": f"{avg_valence:.4f}",
        "Current mood": overall_mood,
        "Dominant Emo": f"{dominant_emotion} ({dominant_score:.2%})",
        "Last Heartbeat": datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    }
    logger.log_stats(f"AGGREGATION: {conversation_id}", stats)
 
    return new_state

async def main():
    await redis_client.connect()
    r = redis_client.redis
    
    # Create consumer group if not exists
    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    logger.info("Aggregation worker started.")
    
    while True:
        try:
            # Read new messages
            streams = {STREAM_KEY: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=1, block=2000)
            
            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        # Process message
                        conversation_id = data.get("conversation_id")
                        original_text = data.get("original_text", "")
                        
                        # Data structure handling:
                        # Central Responder sends: 
                        # emotions: JSON_STRING (map)
                        # dominant_emotion: STRING
                        
                        emotions_json = data.get("emotions", "{}")
                        try:
                            emotions_map = json.loads(emotions_json)
                        except:
                            emotions_map = {}
                            
                        # Add dominant_emotion to the map so update_conversation_state can find it
                        dom_emo = data.get("dominant_emotion")
                        if dom_emo:
                            emotions_map["dominant_emotion"] = dom_emo
                            
                        t0 = time.time()
                        updated_state = await update_conversation_state(conversation_id, new_emotions=emotions_map, r=r, original_text=original_text)
                        elapsed = (time.time() - t0) * 1000
                        logger.debug(f"Aggregation for {conversation_id} took {elapsed:.2f}ms")
                        
                        # Prepare output event
                        output_event = data.copy()
                        output_event["conversation_state"] = json.dumps(updated_state)
                        
                        # Publish to next stage (Persistence / API notification)
                        await redis_client.publish_event(OUTPUT_STREAM, output_event)
                        
                        # Ack message
                        await r.xack(STREAM_KEY, GROUP_NAME, message_id)
            
        except Exception as e:
            logger.log_exception("AGGREGATION WORKER FATAL ERROR", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
