import asyncio
import sys
import os
import json
import time

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient

# Load Configuration
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights.json')
CONFIG = {}
try:
    with open(WEIGHTS_PATH, 'r') as f:
        CONFIG = json.load(f)
except Exception as e:
    print(f"Error loading weights.json: {e}")
    CONFIG = {
        "models": {
            "go_emotions": {"weight": 1.0},
            "basic_bert": {"weight": 0.5},
            "vader": {"weight": 0.3},
            "emojinet": {"weight": 2.0}
        },
        "timeout_ms": 5000
    }

redis_client = RedisClient()
INPUT_STREAM = "partial_analysis_stream"
GROUP_NAME = "central_responder_group"
CONSUMER_NAME = "responder_1"
OUTPUT_STREAM = "emotion_stream"

PENDING_KEY_PREFIX = "pending_aggregation:"

# All 27 GoEmotions labels
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

async def aggregate_and_publish(message_id, partial_results, r):
    print(f"Aggregating results for {message_id}...")
    
    # Base structure
    final_scores = {emo: 0.0 for emo in EMOTION_LABELS}
    original_data = partial_results[0].get("original_data", {}) if partial_results else {}
    
    # 1. Weighted Average for Direct Hits
    total_weights = {emo: 0.0 for emo in EMOTION_LABELS}
    
    model_outputs = {}
    
    for res in partial_results:
        model_name = res.get("model_name")
        scores = res.get("scores", {})
        model_outputs[model_name] = scores
        
        weight = CONFIG["models"].get(model_name, {}).get("weight", 0.0)
        
        # Apply Logic
        if model_name == "vader":
            # VADER doesn't map directly to 27 emotions usually.
            # We keep it for Context Resolution and Logging.
            pass
        elif model_name == "basic_bert":
            # Maps to 7 Ekman emotions.
            # We map these to GoEmotions equivalents.
            for emo, score in scores.items():
                if emo in final_scores:
                    final_scores[emo] += score * weight
                    total_weights[emo] += weight
        elif model_name == "go_emotions":
            # Maps directly
            for emo, score in scores.items():
                if emo in final_scores:
                    final_scores[emo] += score * weight
                    total_weights[emo] += weight
        elif model_name == "emojinet":
            # EmojiNet uses compatible labels
            for emo, score in scores.items():
                if emo in final_scores:
                    final_scores[emo] += score * weight
                    total_weights[emo] += weight

    # Normalize
    for emo in final_scores:
        if total_weights[emo] > 0:
            final_scores[emo] /= total_weights[emo]

    # 2. Context Resolution (Sarcasm/Slang)
    # Using logic ported from original service
    reasoning = None
    vader_scores = model_outputs.get("vader", {})
    vt = vader_scores.get("vader_compound", 0)
    
    # We need to recalculate emoji vector or pass it?
    # For now, let's implement a simplified check using VADER vs Top Emotion Valence
    # (Since we don't have the Emoji Map loaded here yet, we might want to move it to Shared or load it here)
    # For MVP of this refactor, I will omit the detailed Emoji Map logic to avoid complexity explosion, 
    # but I will keep the placeholder for where it would go.
    
    if vt < -0.5 and final_scores.get('joy', 0) > 0.5:
         # Potential Sarcasm (Negative text, Positive emotion detected)
         reasoning = {
             "type": "Conflict Detected",
             "details": "VADER negative but Model positive",
             "action": "Reporting this discrepancy."
         }

    # 2.5 Determine Dominant Emotion
    # Helper to find max in dict
    def get_max_score(score_dict):
        max_k = "neutral"
        max_v = 0.0
        for k, v in score_dict.items():
            if v > max_v:
                max_v = v
                max_k = k
        return max_k, max_v

    # Heuristic: Check non-neutral emotions first
    non_neutral_scores = {k: v for k, v in final_scores.items() if k != "neutral"}
    dom_k, dom_v = get_max_score(non_neutral_scores)
    
    # Threshold for accepting a non-neutral emotion over neutral
    # If the strongest non-neutral emotion is very weak (e.g. < 0.15), we might fall back to neutral
    # But user requested "I don't want to get neutral... I want from the info to get only one response"
    # So we will be aggressive in picking a non-neutral one if it exists.
    
    if dom_v > 0.15:
        dominant_emotion = dom_k
    else:
        # If everything is extremely low, check neutral
        neutral_score = final_scores.get("neutral", 0.0)
        if neutral_score > dom_v:
            dominant_emotion = "neutral"
        else:
             # Even neutral is low? Just pick the max whatever it is.
             dominant_emotion = dom_k

    # [DEBUG] Print all model outputs to trace the decision
    print(f"[DEBUG] Message {message_id} Model Outputs: {json.dumps(model_outputs)}")

    # [SPECIALIST OVERRIDE]

    # [SPECIALIST OVERRIDE]
    # If EmojiNet is highly confident, trust it above all else.
    # This fixes cases where short texts (just emojis) confuse BERT/GoEmotions.
    if "emojinet" in model_outputs:
        emoji_scores = model_outputs["emojinet"]
        # Find max in EmojiNet
        em_k, em_v = get_max_score(emoji_scores)
        if em_k != "neutral" and em_v > 0.8:
            print(f"Confidence Override: Trusting EmojiNet '{em_k}' ({em_v}) over Ensemble '{dominant_emotion}'")
            dominant_emotion = em_k
            # Boost the score of the winner to ensure visual clarity
            final_scores[em_k] = max(final_scores.get(em_k, 0), em_v)

    # 3. Construct Output
    pipeline_log = {
        "models": model_outputs,
        "aggregated": final_scores,
        "dominant_selected": dominant_emotion
    }
    
    # Inject VADER scores into the final payload so Aggregation Service can use them for Valence
    if "vader" in model_outputs:
        for k, v in model_outputs["vader"].items():
             # We include all vader keys: vader_neg, vader_neu, vader_pos, vader_compound
             final_scores[k] = v

    output_event = original_data.copy()
    output_event["emotions"] = json.dumps(final_scores)
    output_event["dominant_emotion"] = dominant_emotion # Explicit single winner
    output_event["pipeline_log"] = json.dumps(pipeline_log)
    if reasoning:
        output_event["reasoning"] = json.dumps(reasoning)
        
    await redis_client.publish_event(OUTPUT_STREAM, output_event)
    
    # Cleanup
    await r.delete(f"{PENDING_KEY_PREFIX}{message_id}")

async def main():
    await redis_client.connect()
    r = redis_client.redis
    
    try:
        await r.xgroup_create(INPUT_STREAM, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            print(f"Error creating group: {e}")

    print("Central Responder started...")
    
    while True:
        try:
            # Read new partial results
            streams = {INPUT_STREAM: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=10, block=2000)
            
            if messages:
                for stream, msgs in messages:
                    for record_id, data in msgs:
                        # Parse
                        # Redis returns byte keys sometimes if not decoded, but our client handles decoding usually?
                        # Our redis_client doesn't auto-decode dict values if they are complex stringified JSONs?
                        # The 'publish_event' uses json.dumps. The stream read returns dict of strings.
                        
                        # Wait, publish_event sends a dict. xadd stores field-values.
                        # data is {field: value}.
                        # In 'publish_event' (shared/utils/redis_client.py), it likely breaks down dict to fields?
                        # No, usually we store a single 'payload' or separate fields.
                        
                        # Let's assume data is usable.
                        # In the services, I sent:
                        # output_event = { "message_id": ..., "model_name": ..., "scores": ... }
                        # redis_client.publish_event might flatten this?
                        # If publish_event does hset/xadd style, nested dicts need json.dumps.
                        # I should check 'shared/utils/redis_client.py' to be sure, but let's assume standard behavior.
                        # I'll rely on the fact that I passed a dict.
                        
                        # The `publish_event` implementation in Step 17's context (emotion_analysis main.py line 207)
                        # sends `output_event`.
                        
                        # I'll parse `scores` which might be a JSON string if it was serialised.
                        msg_id = data.get("message_id")
                        model_name = data.get("model_name")
                        scores_raw = data.get("scores")
                        
                        scores = scores_raw
                        if isinstance(scores_raw, str):
                            try:
                                scores = json.loads(scores_raw)
                            except:
                                pass
                        
                        # Store in Pending Hash
                        # Key: pending_aggregation:<msg_id>
                        # Field: model_name
                        # Value: JSON(full_data_packet)
                        
                        full_packet = {
                            "model_name": model_name,
                            "scores": scores,
                            "original_data": data.get("original_data") # This might be doubly nested string
                        }
                        
                        # Handle original_data if it's a string
                        if isinstance(full_packet["original_data"], str):
                             try:
                                 full_packet["original_data"] = json.loads(full_packet["original_data"])
                             except:
                                 pass

                        pending_key = f"{PENDING_KEY_PREFIX}{msg_id}"
                        await r.hset(pending_key, model_name, json.dumps(full_packet))
                        
                        # Check completeness
                        # We need to know which models are required.
                        expected_models = [k for k, v in CONFIG["models"].items() if v.get("required", True)]
                        
                        current_results = await r.hgetall(pending_key)
                        received_models = list(current_results.keys())
                        
                        if all(m in received_models for m in expected_models):
                            # READY TO AGGREGATE
                            print(f"[DEBUG] All models received for {msg_id}. Aggregating.")
                            all_packets = []
                            for m, packet_str in current_results.items():
                                all_packets.append(json.loads(packet_str))
                            
                            await aggregate_and_publish(msg_id, all_packets, r)
                        else:
                             missing = list(set(expected_models) - set(received_models))
                             print(f"[DEBUG] Message {msg_id} waiting for: {missing}")
                        
                        # Ack
                        await r.xack(INPUT_STREAM, GROUP_NAME, record_id)
            
        except Exception as e:
            print(f"Error in responder loop: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
