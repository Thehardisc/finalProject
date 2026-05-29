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

# Prometheus metrics — /metrics scraped on port 9090
import metrics as METRICS
from prometheus_client import start_http_server as _start_metrics_server

import numpy as np

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from shared.constants import (
    EMOTION_LABELS, CDM_N_STATES,
    FV_CONTEXT_START, FV_DERIVED_START, CTX_CURR_VALENCE,
)

logger = get_logger("central_responder")

redis_client = RedisClient()
INPUT_STREAM = "partial_analysis_stream"
GROUP_NAME = "central_responder_group"
CONSUMER_NAME = "responder_1"
OUTPUT_STREAM = "emotion_stream"
PENDING_KEY_PREFIX = "pending_aggregation:"

AGGREGATION_TIMEOUT_MS = 5000

# context_engine is optional but slower (SentenceTransformer + Qdrant). Once the
# 3 required NLP models arrive, give the CDM context this long to show up before
# aggregating without it (uniform-belief fallback). Tuned for embedding latency.
OPTIONAL_TIMEOUT_MS = float(os.environ.get("OPTIONAL_TIMEOUT_MS", "200"))


def _build_context_snapshot(context, hist_val, resonance, ce_volatility,
                            ce_available, cdm_diag):
    """
    Assemble the context_snapshot forwarded to the frontend.

    Surfaces the PBSM's top-3 simultaneously-active emotion states. When the
    context engine missed the window (no cdm_diag), the snapshot still reports a
    derived top-3 from the conversation's previous dominant emotion so the
    State Machine panel degrades gracefully instead of going blank.
    """
    snap = {
        "prev_emotion":       context["prev_emotion"],
        "avg_valence":        round(context["avg_valence"], 4),
        "historical_valence": round(hist_val, 4),
        "topic_resonance":    round(resonance, 4),
        "volatility":         round(ce_volatility, 4),
        "ce_available":       ce_available,
        "cdm_available":      cdm_diag is not None,
    }

    if cdm_diag:
        snap.update({
            "cdm_state_probs": cdm_diag.get("belief"),
            "cdm_top3_states": cdm_diag.get("top3_states"),
            "cdm_top3_probs":  cdm_diag.get("top3_probs"),
            "cdm_top1_name":   cdm_diag.get("top1_name"),
            "cdm_top2_name":   cdm_diag.get("top2_name"),
            "cdm_top3_name":   cdm_diag.get("top3_name"),
            "cdm_momentum":    cdm_diag.get("momentum"),
            "cdm_residency":   cdm_diag.get("residency"),
            "cdm_catalyst":    cdm_diag.get("catalyst"),
        })

    return snap


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

    # Best-effort: pull the new and previous model metadata from the meta JSON
    # the trainer just wrote so the operator can see the transition delta.
    prev_acc = None
    new_acc  = None
    improvement = None
    try:
        meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "meta_weights_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            new_acc     = meta.get("test_accuracy")
            prev_acc    = meta.get("previous_accuracy")
            improvement = meta.get("improvement")
    except Exception as e:
        logger.warning(
            f"hot_reload_meta_read_failed error={type(e).__name__}",
            extra={"extra_data": {"event": "hot_reload_meta_read_failed", "error": str(e)}},
        )

    META_LEARNER = new_model
    logger.info(
        f"model_hot_reload prev_acc={prev_acc} new_acc={new_acc} improvement={improvement}",
        extra={"extra_data": {
            "event":       "model_hot_reload",
            "prev_acc":    prev_acc,
            "new_acc":     new_acc,
            "improvement": improvement,
        }},
    )

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
    original_data = partial_results[0].get("original_data", {}) if partial_results else {}
    conv_id_early = original_data.get("conversation_id", "conv-1")
    user_id_early = original_data.get("user_id", "")
    mlog = logger.bind(message_id=message_id, conversation_id=conv_id_early, user_id=user_id_early)

    mlog.debug("aggregate_start", extra={"event": "aggregate_start"})

    # format model outputs — separate context_engine from ML models
    model_outputs = {}
    cdm_context_vector = None    # 44-dim CDM context block from context_engine
    cdm_diag           = None    # rich PBSM diagnostics for the frontend
    for res in partial_results:
        model_name = res.get("model_name")
        scores = res.get("scores", {})
        model_outputs[model_name] = scores
        if model_name == "context_engine":
            raw_cv = res.get("context_vector")
            if raw_cv:
                try:
                    cdm_context_vector = json.loads(raw_cv) if isinstance(raw_cv, str) else raw_cv
                except (json.JSONDecodeError, TypeError):
                    cdm_context_vector = None
            raw_diag = res.get("cdm_diag")
            if raw_diag:
                try:
                    cdm_diag = json.loads(raw_diag) if isinstance(raw_diag, str) else raw_diag
                except (json.JSONDecodeError, TypeError):
                    cdm_diag = None

    # Extract emoji scores from go_emotions packet (scored at inference time via GoEmotions model)
    model_outputs["emojinet"] = {}
    for res in partial_results:
        if res.get("model_name") == "go_emotions":
            emoji_scores_raw = res.get("emoji_scores", "{}")
            try:
                model_outputs["emojinet"] = json.loads(emoji_scores_raw) if isinstance(emoji_scores_raw, str) else (emoji_scores_raw or {})
            except (json.JSONDecodeError, TypeError) as e:
                # Silently zeroed emojinet corrupts training samples (one msg has the
                # block, the next doesn't) — surface the parse failure WITHOUT logging
                # raw payload bytes (they can include user-message tokens via emoji shortcodes).
                mlog.warning(
                    "emoji_scores_parse_failed",
                    extra={
                        "event":       "emoji_scores_parse_failed",
                        "error_class": type(e).__name__,
                        "raw_type":    type(emoji_scores_raw).__name__,
                        "raw_len":     len(emoji_scores_raw) if hasattr(emoji_scores_raw, "__len__") else None,
                    },
                )
            break

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
        mlog.warning(
            "conversation_state_fetch_failed",
            extra={"event": "conv_state_fetch_failed", "error": str(e)},
        )

    # Blend EMA valence with user's historical valence for this topic.
    # When topic_resonance is high the user has strong historical feelings about
    # this semantic area — let that shift the effective context by up to 30%.
    if ce_available and resonance > 0.05 and hist_val != 0.0:
        context["avg_valence"] = (
            (1.0 - 0.3 * resonance) * context["avg_valence"]
            + 0.3 * resonance * hist_val
        )
        mlog.debug(
            "context_enriched",
            extra={
                "event":          "context_enriched",
                "resonance":      round(resonance, 4),
                "hist_val":       round(hist_val, 4),
                "effective_val":  round(context["avg_valence"], 4),
            },
        )

    # Inject the 44-dim CDM context block produced by the PBSM. When it didn't
    # arrive in time, build_feature_vector falls back to a uniform belief — the
    # block is never silently zeroed (that was the old 0%-context bug).
    if cdm_context_vector is not None:
        context["cdm_context"] = cdm_context_vector

    mlog.debug(
        "context_injected",
        extra={
            "event":        "context_injected",
            "avg_valence":  round(context["avg_valence"], 4),
            "prev_emotion": context["prev_emotion"],
            "ce_available": ce_available,
            "cdm_available": cdm_context_vector is not None,
        },
    )

    # predict logic
    fv = build_feature_vector(model_outputs, context=context)
    mlog.warning(
        "[DIAG-2-CTX-RECV] "
        f"ce_available={ce_available} cdm_available={cdm_context_vector is not None} "
        f"ctx_block_L2={float(np.linalg.norm(fv[0, FV_CONTEXT_START:FV_DERIVED_START])):.4f} "
        f"meta_loaded={META_LEARNER is not None}"
    )
    dominant_emotion, meta_confidence, final_scores, sarcasm_score, conflict_desc = predict_with_meta_learner(META_LEARNER, fv)
    METRICS.meta_predictions_total.labels(
        outcome="fallback" if dominant_emotion == "neutral" and meta_confidence == 0.0 else "success"
    ).inc()

    # Emoji override: when the meta-learner is uncertain but emoji signal is strong,
    # blend emoji scores in (the model was trained mostly on text without emojis
    # so it under-weights the emoji block until it retrains on emoji-rich data).
    emoji_s = model_outputs.get("emojinet", {})
    if emoji_s and meta_confidence < 0.55:
        top_emoji_label = max(emoji_s, key=emoji_s.get)
        top_emoji_score = emoji_s.get(top_emoji_label, 0.0)
        if top_emoji_score > 0.30:
            blended = {k: final_scores.get(k, 0.0) * 0.5 + emoji_s.get(k, 0.0) * 0.5
                       for k in set(list(final_scores.keys()) + list(emoji_s.keys()))}
            total = sum(blended.values())
            if total > 0:
                final_scores = {k: v / total for k, v in blended.items()}
            dominant_emotion = max(final_scores, key=final_scores.get)
            meta_confidence = final_scores[dominant_emotion]
            mlog.debug(
                "emoji_override_applied",
                extra={"event": "emoji_override", "dominant": dominant_emotion, "confidence": round(meta_confidence, 4)},
            )

    # calculate latency
    # E2E Latency (from ingestion)
    original_ts = original_data.get("timestamp")
    e2e_lat = (time.time() - float(original_ts)) * 1000 if original_ts else 0

    # Log structured inference result (carries bound message_id, conversation_id, user_id).
    top_3 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    mlog.info(
        "meta_inference",
        extra={
            "event":          "meta_inference",
            "dominant":       dominant_emotion,
            "confidence":     round(meta_confidence, 4),
            "top3":           [{"emotion": e, "score": round(s, 4)} for e, s in top_3],
            "avg_valence":    round(context["avg_valence"], 4),
            "prev_emotion":   context["prev_emotion"],
            "e2e_latency_ms": round(e2e_lat, 2),
            "agg_latency_ms": round(agg_lat, 2),
            "conflict":       conflict_desc,
        },
    )

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
        mlog.debug(
            "trajectory_step",
            extra={
                "event":         "trajectory_step",
                "top_predicted": trajectory.get("top_predicted"),
                "top5":          list(trajectory.get("predicted_next", {}).keys()),
            },
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
        "context_snapshot":  _build_context_snapshot(
            context, hist_val, resonance, ce_volatility, ce_available, cdm_diag
        ),
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

    METRICS.messages_processed_total.inc()
    METRICS.aggregation_latency_ms.observe(agg_lat)

    # Cleanup
    await r.delete(f"{PENDING_KEY_PREFIX}{message_id}")

async def main():
    # Start Prometheus /metrics server BEFORE connecting to Redis so it's
    # scrapeable even during a slow startup (e.g., model retraining cycle).
    try:
        _start_metrics_server(9090)
        logger.info("Prometheus metrics server listening on :9090/metrics")
    except OSError as e:
        logger.warning(f"Could not start metrics server on :9090: {e}")

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
                        rlog = logger.bind(message_id=msg_id, model=model_name)

                        scores = scores_raw
                        if isinstance(scores_raw, str):
                            try:
                                scores = json.loads(scores_raw)
                            except (json.JSONDecodeError, TypeError) as e:
                                rlog.warning(
                                    "scores_parse_failed",
                                    extra={"event": "scores_parse_failed", "error_class": type(e).__name__},
                                )

                        # Store in Pending Hash
                        # Key: pending_aggregation:<msg_id>
                        # Field: model_name
                        # Value: JSON(full_data_packet)

                        full_packet = {
                            "model_name": model_name,
                            "scores": scores,
                            "original_data": data.get("original_data"), # This might be doubly nested string
                            "emoji_scores": data.get("emoji_scores", "{}"),
                            # CDM context block + diagnostics from context_engine.
                            # Preserved here so they survive to aggregate_and_publish.
                            "context_vector": data.get("context_vector"),
                            "cdm_diag": data.get("cdm_diag"),
                        }

                        # Handle original_data if it's a string
                        if isinstance(full_packet["original_data"], str):
                             try:
                                 full_packet["original_data"] = json.loads(full_packet["original_data"])
                             except (json.JSONDecodeError, TypeError) as e:
                                 rlog.warning(
                                     "original_data_parse_failed",
                                     extra={"event": "original_data_parse_failed", "error_class": type(e).__name__},
                                 )

                        pending_key = f"{PENDING_KEY_PREFIX}{msg_id}"

                        # setup pending object
                        # add expiration
                        is_new_key = not await r.exists(pending_key)
                        if is_new_key:
                            await r.hset(pending_key, "arrival_timestamp", time.time())

                        await r.hset(pending_key, model_name, json.dumps(full_packet))
                        await r.expire(pending_key, 30)  # refresh TTL on every update

                        expected_models = ["go_emotions", "basic_bert", "vader"]

                        current_results = await r.hgetall(pending_key)
                        received_models = [k for k in current_results.keys() if k != "arrival_timestamp"]

                        required_ready = all(m in received_models for m in expected_models)

                        # Once the required models are in, give the slower
                        # context_engine (SentenceTransformer + Qdrant) a bounded
                        # inline grace window so CDM context isn't perpetually lost
                        # to the race. The wait is capped and always falls through to
                        # aggregation, so a message never gets stranded if context
                        # never arrives (build_feature_vector uses uniform fallback).
                        if required_ready and "context_engine" not in received_models:
                            deadline = time.time() + OPTIONAL_TIMEOUT_MS / 1000.0
                            while time.time() < deadline:
                                await asyncio.sleep(0.025)
                                current_results = await r.hgetall(pending_key)
                                if "context_engine" in current_results:
                                    break
                            received_models = [k for k in current_results.keys() if k != "arrival_timestamp"]

                        if required_ready:
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
