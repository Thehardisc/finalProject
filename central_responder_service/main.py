import asyncio
import sys
import os
import json

# Meta-learner module (safe — returns None on any load failure)
from meta_learner import (
    load_meta_learner,
    build_feature_vector,
    predict_with_meta_learner,
)

# Periodic retraining daemon (runs as a background thread)
from trainer import start_trainer_thread

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient

redis_client = RedisClient()
INPUT_STREAM = "partial_analysis_stream"
GROUP_NAME = "central_responder_group"
CONSUMER_NAME = "responder_1"
OUTPUT_STREAM = "emotion_stream"
PENDING_KEY_PREFIX = "pending_aggregation:"

AGGREGATION_TIMEOUT_MS = 5000

# ── Meta-learner startup load ─────────────────────────────────────────────────
# The trained .pkl from models/ is STRICTLY REQUIRED.
# If missing, we crash immediately so the user knows they must run training.
META_LEARNER = load_meta_learner()
if META_LEARNER is None:
    print("[CentralResponder] ❌ CRITICAL: No meta_weights.pkl found! You MUST run the training script first.")
    sys.exit(1)

print("[CentralResponder] ✅  Running in META-LEARNER mode.")

# ── Hot-reload callback ─────────────────────────────────────────────────
# Called by the trainer thread after a successful deploy.
# Swaps the in-memory model immediately — no container restart needed.
def on_model_reload(new_model):
    global META_LEARNER
    META_LEARNER = new_model
    print("[CentralResponder] 🔄 Meta-learner hot-reloaded in memory.")

# ── Start periodic retraining thread ────────────────────────────────────
start_trainer_thread(on_model_reload)
print("[CentralResponder] ℹ️ Background meta-learner retraining daemon is ACTIVATED with transient auto-garbage collection.")


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
    
    original_data = partial_results[0].get("original_data", {}) if partial_results else {}
    
    # ── 1. Gather scores from analyzers ───────────────────────────────────────
    model_outputs = {}
    for res in partial_results:
        model_name = res.get("model_name")
        scores = res.get("scores", {})
        model_outputs[model_name] = scores

    # ── 2. Meta-Learner Inference ─────────────────────────────────────────────
    # The meta-learner is mandatory. It consumes all 4 model outputs and emits
    # the dominant emotion + a full probability distribution over all 27 classes.
    try:
        fv = build_feature_vector(model_outputs)
        dominant_emotion, meta_confidence, final_scores = predict_with_meta_learner(META_LEARNER, fv)
        print(f"[MetaLearner] Prediction: '{dominant_emotion}' (confidence={meta_confidence:.3f})")
    except Exception as e:
        print(f"[MetaLearner] ⚠️ Predict failed ({e}). Defaulting to neutral.")
        dominant_emotion = "neutral"
        meta_confidence = 0.0
        final_scores = {emo: 0.0 for emo in EMOTION_LABELS}

    # ── 3. Sarcasm / Conflict Check (Optional reasoning layer) ────────────────
    reasoning = None
    vt = model_outputs.get("vader", {}).get("vader_compound", 0)
    
    if vt < -0.5 and final_scores.get('joy', 0) > 0.5:
         reasoning = {
             "type": "Conflict Detected",
             "details": "VADER negative but MetaLearner predicts Joy",
             "action": "Reporting this discrepancy."
         }

    # ── 4. Construct Output ───────────────────────────────────────────────────
    pipeline_log = {
        "models": model_outputs,
        "aggregated": final_scores,
        "dominant_selected": dominant_emotion,
        "decision_mode": "meta-learner",
        "meta_confidence": meta_confidence,
    }
    
    # Inject VADER scores into the final payload for the Aggregation/Frontend service
    if "vader" in model_outputs:
        for k, v in model_outputs["vader"].items():
             final_scores[k] = v

    output_event = original_data.copy()
    output_event["emotions"] = json.dumps(final_scores)
    output_event["dominant_emotion"] = dominant_emotion 
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
                        
                        # We need to know which models are required before aggregating.
                        expected_models = ["go_emotions", "basic_bert", "vader", "emojinet"]
                        
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
