import asyncio
import sys
import os
import json
import time

# Add parent directory to path to import shared (must come before shared imports)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import meta learner
from meta_learner import (
    load_meta_learner,
    build_feature_vector,
    predict_with_meta_learner,
    calculate_feature_impacts
)

# import background trainer
from trainer import start_trainer_thread

# import trajectory LSTM
from trajectory.inference import load_trajectory_model, run_trajectory_step

import emoji as emoji_lib
from shared.constants import EMOJI_EMOTION_DB

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger

logger = get_logger("central_responder")

redis_client = RedisClient()
INPUT_STREAM = "partial_analysis_stream"
GROUP_NAME = "central_responder_group"
CONSUMER_NAME = "responder_1"
OUTPUT_STREAM = "emotion_stream"
PENDING_KEY_PREFIX = "pending_aggregation:"

AGGREGATION_TIMEOUT_MS = 5000

# load weights
# requires the pkl file
META_LEARNER = load_meta_learner()
if META_LEARNER is None:
    logger.warning("No compatible meta_weights.pkl found. Falling back to Rule-Based Aggregation until trainer finishes.")
else:
    logger.info("Running in META-LEARNER mode.")

# update model when retrained
def on_model_reload(new_model):
    global META_LEARNER
    META_LEARNER = new_model
    logger.info("Meta-learner hot-reloaded in memory.")

# check model status for gate
ready_marker = os.path.join(os.path.dirname(__file__), "..", "models", ".ready")

if META_LEARNER is not None:
    # model ok - write ready file
    open(ready_marker, 'w').close()
    logger.info("✅ [GATEKEEPER] Valid model found. Gate is OPEN. Background retraining will run periodically.")
else:
    # missing model - remove ready file
    if os.path.exists(ready_marker):
        try:
            os.remove(ready_marker)
        except:
            pass
    logger.info("No valid model found. Missing ready file.")

start_trainer_thread(on_model_reload)
logger.info("Background meta-learner retraining daemon is ACTIVATED with transient auto-garbage collection.")

# load trajectory LSTM (optional — degrades gracefully if missing)
_model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
TRAJECTORY_MODEL = load_trajectory_model(
    model_path=os.path.join(_model_dir, 'trajectory_lstm.pt'),
    config_path=os.path.join(_model_dir, 'trajectory_config.json'),
)
if TRAJECTORY_MODEL is not None:
    logger.info("Trajectory LSTM loaded — conversation direction prediction ACTIVE.")


# All 27 GoEmotions labels
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def _emojinet_inline(text: str) -> dict:
    """Compute emoji emotion scores inline — no network hop needed."""
    try:
        found  = emoji_lib.distinct_emoji_list(text)
        scores = {}
        count  = 0
        for ch in found:
            entry = EMOJI_EMOTION_DB.get(ch) or EMOJI_EMOTION_DB.get(ch.replace('️', ''))
            if entry:
                for emo, sc in entry.get("emotions", {}).items():
                    scores[emo] = scores.get(emo, 0.0) + sc
                count += 1
        if count:
            result = {k: v / count for k, v in scores.items()}
            result.setdefault("neutral", 0.0)
            return result
    except Exception as e:
        logger.warning(f"Inline emojinet error: {e}")
    return {}


async def aggregate_and_publish(message_id, partial_results, r, agg_lat=0):
    logger.debug(f"Aggregating results for {message_id}...")
    
    original_data = partial_results[0].get("original_data", {}) if partial_results else {}
    
    # format model outputs — separate context_engine from ML models
    model_outputs = {}
    for res in partial_results:
        model_name = res.get("model_name")
        scores = res.get("scores", {})
        model_outputs[model_name] = scores

    # Extract context_engine enrichment (optional — may not arrive before the 3 ML models)
    ce            = model_outputs.pop("context_engine", {})
    hist_val      = float(ce.get("historical_valence", 0.0))
    resonance     = float(ce.get("topic_resonance",    0.0))
    ce_volatility = float(ce.get("volatility",         0.0))
    ce_available  = bool(ce)

    # get context from previous message's aggregation result
    conv_id = original_data.get("conversation_id", "conv-1")
    context = {"avg_valence": 0.0, "prev_emotion": "neutral"}
    try:
        state_key = f"conversation:{conv_id}"
        state = await r.hgetall(state_key)
        if state:
            context["avg_valence"]  = float(state.get("average_valence", 0.0))
            context["prev_emotion"] = state.get("dominant_emotion", "neutral")
    except Exception as e:
        logger.warning(f"Failed to fetch context for {conv_id}: {e}")

    # Blend EMA valence with user's historical valence for this topic.
    # When topic_resonance is high the user has strong historical feelings about
    # this semantic area — let that shift the effective context by up to 30%.
    if ce_available and resonance > 0.05 and hist_val != 0.0:
        context["avg_valence"] = (
            (1.0 - 0.3 * resonance) * context["avg_valence"]
            + 0.3 * resonance * hist_val
        )
        logger.debug(
            f"Context enriched by episodic memory: resonance={resonance:.2f}, "
            f"hist_val={hist_val:.3f} → effective_val={context['avg_valence']:.3f}"
        )

    logger.debug(
        f"Context injected for {conv_id}: "
        f"val={context['avg_valence']:.3f}, mood={context['prev_emotion']}, "
        f"ce={'yes' if ce_available else 'no'}"
    )

    # predict logic
    fv = build_feature_vector(model_outputs, context=context)
    dominant_emotion, meta_confidence, final_scores, sarcasm_score, conflict_desc = predict_with_meta_learner(META_LEARNER, fv)

    # calculate latency
    # E2E Latency (from ingestion)
    original_ts = original_data.get("timestamp")
    e2e_lat = (time.time() - float(original_ts)) * 1000 if original_ts else 0

    # Log Stats Block
    top_3 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    stats = {
        "Message ID": message_id,
        "Dominant Emotion": f"'{dominant_emotion}' ({meta_confidence:.2%})",
        "Context Injected": f"Val:{context['avg_valence']:.2f}, Mood:{context['prev_emotion']}",
        "Top 3 Probabilities": ", ".join([f"{e}:{s:.2%}" for e, s in top_3]),
        "E2E Latency": f"{e2e_lat:.2f}ms",
        "Agg Latency": f"{agg_lat:.2f}ms"
    }
    if conflict_desc:
        stats["Conflict"] = conflict_desc

    logger.log_stats(f"Meta-Inference: {message_id}", stats)

    # evaluate sarcasm
    reasoning = None
    if conflict_desc or sarcasm_score > 0.3:
         reasoning = {
             "type": "Contextual Dissonance" if conflict_desc else "Sarcastic Intent",
             "details": conflict_desc or "Emotional signal flip detected.",
             "sarcasm_score": float(sarcasm_score),
             "action": "Meta-Learner nuance override applied."
         }

    # Calculate Logic Map (Impact Scores)
    logic_map = calculate_feature_impacts(META_LEARNER, fv, dominant_emotion)

    # Run trajectory LSTM step — updates conversation hidden state in Redis,
    # returns predicted next-message emotion distribution + trajectory embedding
    trajectory = await run_trajectory_step(
        model=TRAJECTORY_MODEL,
        model_outputs=model_outputs,
        conv_id=conv_id,
        redis=r,
    )
    if trajectory:
        logger.debug(
            f"Trajectory: next predicted='{trajectory.get('top_predicted')}' "
            f"| top-5={list(trajectory.get('predicted_next', {}).keys())}"
        )

    # format output
    pipeline_log = {
        "models":            model_outputs,
        "aggregated":        final_scores,
        "dominant_selected": dominant_emotion,
        "decision_mode":     "meta-learner",
        "meta_confidence":   meta_confidence,
        "logic_map":         logic_map,
        "sarcasm_score":     float(sarcasm_score),
        "conflict":          conflict_desc,
        "trajectory":        trajectory,
        "context_snapshot":  {
            "prev_emotion":       context["prev_emotion"],
            "avg_valence":        round(context["avg_valence"], 4),
            "historical_valence": round(hist_val, 4),
            "topic_resonance":    round(resonance, 4),
            "volatility":         round(ce_volatility, 4),
            "ce_available":       ce_available,
        },
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
        
    # Context Divergence Detection (Bonus)
    prev_mood = context.get("prev_emotion", "neutral")
    if prev_mood != "neutral" and dominant_emotion != prev_mood:
        divergence = {
            "type": "Context Shift",
            "from": prev_mood,
            "to": dominant_emotion,
            "significance": "High" if meta_confidence > 0.7 else "Moderate"
        }
        output_event["context_shift"] = json.dumps(divergence)

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
            logger.error(f"Error creating group: {e}")

    logger.info("Central Responder started.")
    
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
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.warning(f"Cannot parse scores for model '{model_name}' (msg {msg_id}): {e}")
                        
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

                        # setup pending object
                        # add expiration
                        is_new_key = not await r.exists(pending_key)
                        if is_new_key:
                            await r.hset(pending_key, "arrival_timestamp", time.time())

                        await r.hset(pending_key, model_name, json.dumps(full_packet))
                        await r.expire(pending_key, 30)  # refresh TTL on every update

                        # emojinet computed inline below; context read from Redis in aggregate_and_publish
                        expected_models = ["go_emotions", "basic_bert", "vader"]

                        current_results = await r.hgetall(pending_key)
                        received_models = [k for k in current_results.keys() if k != "arrival_timestamp"]

                        if all(m in received_models for m in expected_models):
                            logger.debug(f"All models received for {msg_id}. Aggregating.")

                            all_packets = []
                            arrival_ts = float(current_results.get("arrival_timestamp", time.time()))
                            agg_lat = (time.time() - arrival_ts) * 1000

                            for m, packet_str in current_results.items():
                                if m == "arrival_timestamp":
                                    continue
                                try:
                                    all_packets.append(json.loads(packet_str))
                                except Exception:
                                    logger.error(f"Failed to parse packet for model {m}")

                            # Compute emojinet inline and inject as synthetic packet
                            first_original = next(
                                (p.get("original_data", {}) for p in all_packets if p.get("original_data")), {}
                            )
                            raw_text = first_original.get("text", "") or first_original.get("original_text", "")
                            emoji_scores = _emojinet_inline(raw_text)
                            all_packets.append({
                                "model_name":    "emojinet",
                                "scores":        emoji_scores,
                                "original_data": first_original,
                            })

                            try:
                                await aggregate_and_publish(msg_id, all_packets, r, agg_lat=agg_lat)
                            except Exception as agg_err:
                                logger.error(f"[AGGREGATION FAILED] {msg_id}: {agg_err}")
                        else:
                            missing = list(set(expected_models) - set(received_models))
                            logger.debug(f"Message {msg_id} waiting for: {missing}")

                        # ACK the stream record regardless of outcome
                        await r.xack(INPUT_STREAM, GROUP_NAME, record_id)

        except Exception as e:
            logger.log_exception("CENTRAL RESPONDER CRITICAL ERROR", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
