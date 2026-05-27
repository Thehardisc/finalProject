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
    calculate_feature_impacts,
    apply_context_correction,
)

# import background trainer
from trainer import start_trainer_thread

# import trajectory LSTM
from trajectory.inference import load_trajectory_model, run_trajectory_step


from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from shared.constants import (
    CONTEXT_DIM,
    CTX_HIST_VALENCE, CTX_RESONANCE, CTX_VOLATILITY, CTX_CURR_VALENCE,
)

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


async def aggregate_and_publish(message_id, partial_results, r, agg_lat=0):
    logger.debug(f"Aggregating results for {message_id}...")

    original_data = partial_results[0].get("original_data", {}) if partial_results else {}

    # Separate context_engine (151-dim CDM vector) from ML model score dicts
    model_outputs = {}
    context_vector = None
    for res in partial_results:
        model_name = res.get("model_name")
        if model_name == "context_engine":
            raw_cv = res.get("context_vector")
            if raw_cv:
                try:
                    context_vector = json.loads(raw_cv) if isinstance(raw_cv, str) else raw_cv
                except (json.JSONDecodeError, TypeError):
                    context_vector = None
        else:
            model_outputs[model_name] = res.get("scores", {})

    ce_available = context_vector is not None and len(context_vector) == CONTEXT_DIM
    if not ce_available:
        received_dim = len(context_vector) if context_vector else 0
        if received_dim > 0:
            logger.warning(
                f"[SCHEMA MISMATCH] context_vector dim={received_dim}, "
                f"expected {CONTEXT_DIM} (CONTEXT_DIM in shared/constants.py). "
                f"Injecting zeros — check context_engine_service version."
            )
        context_vector = [0.0] * CONTEXT_DIM

    # Fetch prev_emotion from Redis for context-shift detection
    conv_id = original_data.get("conversation_id", "conv-1")
    prev_emotion = "neutral"
    try:
        state = await r.hgetall(f"conversation:{conv_id}")
        if state:
            prev_emotion = state.get("dominant_emotion", "neutral")
    except Exception as e:
        logger.warning(f"Failed to fetch prev_emotion for {conv_id}: {e}")

    logger.debug(
        f"Context for {conv_id}: prev_emotion={prev_emotion}, "
        f"ce={'yes' if ce_available else 'no (zeros)'}"
    )

    # Predict
    fv = build_feature_vector(model_outputs, context_vector=context_vector)
    dominant_emotion, meta_confidence, final_scores, sarcasm_score, conflict_desc = predict_with_meta_learner(META_LEARNER, fv)

    # Context correction — additive blend toward valence-consistent emotions
    corrected_scores, ctx_correction_weight = apply_context_correction(
        final_scores, context_vector, prev_emotion=prev_emotion
    )
    if ctx_correction_weight > 0:
        final_scores     = corrected_scores
        dominant_emotion = max(final_scores, key=final_scores.get)
        meta_confidence  = final_scores[dominant_emotion]

    # E2E Latency
    original_ts = original_data.get("timestamp")
    e2e_lat = (time.time() - float(original_ts)) * 1000 if original_ts else 0

    # Prefer named scalars published by context_engine (no index guessing).
    # Fall back to indexed access only if the named fields are absent (old version).
    _ce_packet = next((r for r in partial_results if r.get("model_name") == "context_engine"), {})
    def _named(key, idx):
        v = _ce_packet.get(key)
        return round(float(v), 4) if v is not None else round(float(context_vector[idx]), 4)

    hist_val   = _named("ctx_hist_valence",    CTX_HIST_VALENCE)
    resonance  = _named("ctx_topic_resonance", CTX_RESONANCE)
    volatility = _named("ctx_volatility",      CTX_VOLATILITY)
    cur_val    = _named("ctx_curr_valence",    CTX_CURR_VALENCE)

    top_3 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    stats = {
        "Message ID":        message_id,
        "Dominant Emotion":  f"'{dominant_emotion}' ({meta_confidence:.2%})",
        "Prev Emotion":      prev_emotion,
        "CE Context":        f"cur:{cur_val:.3f} hist:{hist_val:.3f} vol:{volatility:.3f} ce={'yes' if ce_available else 'no'}",
        "Top 3":             ", ".join([f"{e}:{s:.2%}" for e, s in top_3]),
        "E2E Latency":       f"{e2e_lat:.2f}ms",
        "Agg Latency":       f"{agg_lat:.2f}ms",
    }
    if conflict_desc:
        stats["Conflict"] = conflict_desc
    logger.log_stats(f"Meta-Inference: {message_id}", stats)

    reasoning = None
    if conflict_desc or sarcasm_score > 0.3:
        reasoning = {
            "type":         "Contextual Dissonance" if conflict_desc else "Sarcastic Intent",
            "details":      conflict_desc or "Emotional signal flip detected.",
            "sarcasm_score": float(sarcasm_score),
            "action":        "Meta-Learner nuance override applied.",
        }

    logic_map = calculate_feature_impacts(META_LEARNER, fv, dominant_emotion)

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

    cdm_probs = context_vector[0:7] if ce_available else [0.0] * 7
    cdm_state_idx = cdm_probs.index(max(cdm_probs)) if ce_available else None

    pipeline_log = {
        "models":                model_outputs,
        "aggregated":            final_scores,
        "dominant_selected":     dominant_emotion,
        "decision_mode":         "meta-learner",
        "meta_confidence":       meta_confidence,
        "logic_map":             logic_map,
        "ctx_correction_weight": ctx_correction_weight,
        "sarcasm_score":         float(sarcasm_score),
        "conflict":              conflict_desc,
        "trajectory":            trajectory,
        "context_snapshot":  {
            "prev_emotion":         prev_emotion,
            "cur_valence":          cur_val,
            "historical_valence":   hist_val,
            "topic_resonance":      resonance,
            "volatility":           volatility,
            "ce_available":         ce_available,
            "cdm_state_probs":      [round(float(p), 4) for p in cdm_probs],
            "cdm_current_state":    cdm_state_idx,
            "cdm_residency":        round(float(context_vector[7]), 3) if ce_available else 0.0,
            "cdm_entry_abruptness": round(float(context_vector[11]), 3) if ce_available else 0.0,
            "cdm_available":        ce_available,
        },
    }

    # Inject VADER scores into the final payload for the Aggregation/Frontend service
    if "vader" in model_outputs:
        for k, v in model_outputs["vader"].items():
            final_scores[k] = v

    output_event = original_data.copy()
    output_event["emotions"]         = json.dumps(final_scores)
    output_event["dominant_emotion"] = dominant_emotion
    output_event["pipeline_log"]     = json.dumps(pipeline_log)
    if reasoning:
        output_event["reasoning"] = json.dumps(reasoning)

    # Context Divergence Detection
    if prev_emotion != "neutral" and dominant_emotion != prev_emotion:
        divergence = {
            "type":         "Context Shift",
            "from":         prev_emotion,
            "to":           dominant_emotion,
            "significance": "High" if meta_confidence > 0.7 else "Moderate",
        }
        output_event["context_shift"] = json.dumps(divergence)

    await redis_client.publish_event(OUTPUT_STREAM, output_event)
    
    # Cleanup is handled by the in-memory dict caller

pending_aggregations = {}

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
            # PEL recovery: reclaim partial results idle >30s (handles crash-before-ACK)
            try:
                _, stale, _ = await r.xautoclaim(INPUT_STREAM, GROUP_NAME, CONSUMER_NAME, min_idle_time=30_000, start_id='0-0', count=10)
                if stale:
                    logger.info(f"[PEL] Reclaimed {len(stale)} stale message(s) from partial_analysis_stream")
                    messages = [(INPUT_STREAM, stale)]
                else:
                    messages = None
            except Exception:
                messages = None

            if not messages:
                messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, {INPUT_STREAM: ">"}, count=10, block=2000)

            if messages:
                for stream, msgs in messages:
                     for record_id, data in msgs:
                        msg_id = data.get("message_id")
                        model_name = data.get("model_name")

                        # context_engine publishes context_vector + named scalars
                        if model_name == "context_engine":
                            full_packet = {
                                "model_name":          model_name,
                                "context_vector":      data.get("context_vector"),
                                "original_data":       data.get("original_data"),
                                # Named scalars — read by aggregate_and_publish without index guessing
                                "ctx_hist_valence":    data.get("ctx_hist_valence"),
                                "ctx_topic_resonance": data.get("ctx_topic_resonance"),
                                "ctx_volatility":      data.get("ctx_volatility"),
                                "ctx_curr_valence":    data.get("ctx_curr_valence"),
                                "ctx_dim":             data.get("ctx_dim"),
                            }
                        else:
                            scores_raw = data.get("scores")
                            scores = scores_raw
                            if isinstance(scores_raw, str):
                                try:
                                    scores = json.loads(scores_raw)
                                except (json.JSONDecodeError, TypeError) as e:
                                    logger.warning(f"Cannot parse scores for model '{model_name}' (msg {msg_id}): {e}")
                            full_packet = {
                                "model_name":    model_name,
                                "scores":        scores,
                                "original_data": data.get("original_data"),
                            }

                        # Handle original_data if it's a string
                        if isinstance(full_packet["original_data"], str):
                            try:
                                full_packet["original_data"] = json.loads(full_packet["original_data"])
                            except Exception:
                                pass

                        if msg_id not in pending_aggregations:
                            pending_aggregations[msg_id] = {"arrival_timestamp": time.time(), "models": {}}
                        
                        pending_aggregations[msg_id]["models"][model_name] = full_packet

                        # Publish partial result for streaming pipeline progress in the frontend
                        if model_name != "context_engine":
                            orig = full_packet.get("original_data")
                            conv_id = orig.get("conversation_id") if isinstance(orig, dict) else None
                            if conv_id:
                                try:
                                    await redis_client.publish_event("partial_result_stream", {
                                        "message_id":      msg_id,
                                        "conversation_id": conv_id,
                                        "model":           model_name,
                                    })
                                except Exception:
                                    pass

                        expected_models = ["go_emotions", "basic_bert", "vader"]

                        current_results = pending_aggregations[msg_id]["models"]
                        received_models = list(current_results.keys())

                        if all(m in received_models for m in expected_models):
                            logger.debug(f"All models received for {msg_id}. Aggregating.")

                            all_packets = list(current_results.values())
                            arrival_ts = pending_aggregations[msg_id]["arrival_timestamp"]
                            agg_lat = (time.time() - arrival_ts) * 1000

                            try:
                                await aggregate_and_publish(msg_id, all_packets, r, agg_lat=agg_lat)
                            except Exception as agg_err:
                                logger.error(f"[AGGREGATION FAILED] {msg_id}: {agg_err}")
                            
                            # Cleanup
                            del pending_aggregations[msg_id]
                        else:
                            missing = list(set(expected_models) - set(received_models))
                            logger.debug(f"Message {msg_id} waiting for: {missing}")

                        # ACK the stream record regardless of outcome
                        await r.xack(INPUT_STREAM, GROUP_NAME, record_id)
                        
            # Cleanup stale pending aggregations older than 60 seconds
            current_time = time.time()
            stale_keys = [msg_id for msg_id, info in pending_aggregations.items() if current_time - info["arrival_timestamp"] > 60]
            for stale_key in stale_keys:
                del pending_aggregations[stale_key]

        except Exception as e:
            logger.log_exception("CENTRAL RESPONDER CRITICAL ERROR", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
