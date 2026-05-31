"""
core/aggregator.py — Collect partial model results and publish the final prediction.

This is the single source of truth for fusing the 3 ML model partials (+ emoji +
context engine) into one emotion event. Mirrors what used to live inline in
main.py:aggregate_and_publish, now hardened (bound logging, rule-based fallback,
explicit decision_mode) and modular.
"""
import json
import time

from shared.utils.logger import get_logger
from shared.utils.redis_client import RedisClient

# Absolute imports (service-dir rooted) so this is importable under `python main.py`.
from ml.predictor         import build_feature_vector, predict_with_meta_learner, rule_based_predict
from ml.impact_calculator  import calculate_feature_impacts
from trajectory.inference  import run_trajectory_step

import metrics as METRICS

logger = get_logger("central_responder")

OUTPUT_STREAM      = "emotion_stream"
PENDING_KEY_PREFIX = "pending_aggregation:"

# Emoji-override tuning — the meta-learner under-weights the emoji block until it
# retrains on emoji-rich data, so when it's uncertain and emoji signal is strong we
# blend the emoji distribution in.
EMOJI_OVERRIDE_CONF_CEILING = 0.55   # only override when meta confidence is below this
EMOJI_OVERRIDE_MIN_SCORE    = 0.30   # and the top emoji score clears this
EMOJI_BLEND_WEIGHT          = 0.5    # 50/50 blend of meta vs emoji


async def aggregate_and_publish(message_id, partial_results, meta_learner,
                                 trajectory_model, redis_client: RedisClient,
                                 r, agg_lat: float = 0) -> None:
    """
    Build the 103-dim feature vector from all model results, run the meta-learner
    (or rule-based fallback), apply emoji override + context enrichment + trajectory,
    and publish the enriched result to emotion_stream.
    """
    original_data = partial_results[0].get("original_data", {}) if partial_results else {}
    conv_id       = original_data.get("conversation_id", "conv-1")
    user_id       = original_data.get("user_id", "")
    mlog = logger.bind(message_id=message_id, conversation_id=conv_id, user_id=user_id)

    mlog.debug("aggregate_start", extra={"event": "aggregate_start"})

    # ── Collect model outputs ────────────────────────────────────────────────
    model_outputs = {}
    for res in partial_results:
        model_outputs[res.get("model_name")] = res.get("scores", {})

    # Extract emoji scores from the go_emotions packet (scored upstream via GoEmotions).
    model_outputs["emojinet"] = {}
    for res in partial_results:
        if res.get("model_name") == "go_emotions":
            emoji_scores_raw = res.get("emoji_scores", "{}")
            try:
                model_outputs["emojinet"] = (
                    json.loads(emoji_scores_raw) if isinstance(emoji_scores_raw, str)
                    else (emoji_scores_raw or {})
                )
            except (json.JSONDecodeError, TypeError) as e:
                # Surface the parse failure WITHOUT logging raw bytes (they can carry
                # user-message tokens via emoji shortcodes). A silently-zeroed emojinet
                # block would corrupt the train/inference distribution.
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

    # ── Context-engine enrichment (optional — may not arrive before the 3 ML models) ──
    ce            = model_outputs.pop("context_engine", {})
    hist_val      = float(ce.get("historical_valence", 0.0))
    resonance     = float(ce.get("topic_resonance",    0.0))
    ce_volatility = float(ce.get("volatility",         0.0))
    ce_available  = bool(ce)

    # ── Conversation context from previous message's aggregation result ──────────
    context = {"avg_valence": 0.0, "prev_emotion": "neutral"}
    try:
        state = await r.hgetall(f"conversation:{conv_id}")
        if state:
            context["avg_valence"]  = float(state.get("average_valence", 0.0))
            context["prev_emotion"] = state.get("dominant_emotion", "neutral")
    except Exception as e:
        mlog.warning(
            "conversation_state_fetch_failed",
            extra={"event": "conv_state_fetch_failed", "error": str(e)},
        )

    # Blend EMA valence with the user's historical valence for this topic. High
    # topic_resonance means strong historical feelings — shift the effective
    # context by up to 30%.
    if ce_available and resonance > 0.05 and hist_val != 0.0:
        context["avg_valence"] = (
            (1.0 - 0.3 * resonance) * context["avg_valence"]
            + 0.3 * resonance * hist_val
        )
        mlog.debug(
            "context_enriched",
            extra={
                "event":         "context_enriched",
                "resonance":     round(resonance, 4),
                "hist_val":      round(hist_val, 4),
                "effective_val": round(context["avg_valence"], 4),
            },
        )

    mlog.debug(
        "context_injected",
        extra={
            "event":        "context_injected",
            "avg_valence":  round(context["avg_valence"], 4),
            "prev_emotion": context["prev_emotion"],
            "ce_available": ce_available,
        },
    )

    # ── Inference (meta-learner, or genuine rule-based fallback) ─────────────────
    fv = build_feature_vector(model_outputs, context=context)
    if meta_learner is None:
        dominant_emotion, meta_confidence, final_scores, sarcasm_score, conflict_desc = \
            rule_based_predict(model_outputs)
        decision_mode = "rule-based"
    else:
        dominant_emotion, meta_confidence, final_scores, sarcasm_score, conflict_desc = \
            predict_with_meta_learner(meta_learner, fv)
        # predict_with_meta_learner returns this exact sentinel only when it failed.
        if dominant_emotion == "neutral" and meta_confidence == 0.0 and not final_scores:
            dominant_emotion, meta_confidence, final_scores, sarcasm_score, conflict_desc = \
                rule_based_predict(model_outputs)
            decision_mode = "rule-based"
        else:
            decision_mode = "meta-learner"

    METRICS.meta_predictions_total.labels(
        outcome="fallback" if decision_mode == "rule-based" else "success"
    ).inc()

    # ── Emoji override: blend strong emoji signal when the meta-learner is unsure ──
    emoji_s = model_outputs.get("emojinet", {})
    if emoji_s and meta_confidence < EMOJI_OVERRIDE_CONF_CEILING:
        top_emoji_label = max(emoji_s, key=emoji_s.get)
        top_emoji_score = emoji_s.get(top_emoji_label, 0.0)
        if top_emoji_score > EMOJI_OVERRIDE_MIN_SCORE:
            w = EMOJI_BLEND_WEIGHT
            blended = {k: final_scores.get(k, 0.0) * (1 - w) + emoji_s.get(k, 0.0) * w
                       for k in set(list(final_scores.keys()) + list(emoji_s.keys()))}
            total = sum(blended.values())
            if total > 0:
                final_scores = {k: v / total for k, v in blended.items()}
            dominant_emotion = max(final_scores, key=final_scores.get)
            meta_confidence  = final_scores[dominant_emotion]
            mlog.debug(
                "emoji_override_applied",
                extra={"event": "emoji_override", "dominant": dominant_emotion,
                       "confidence": round(meta_confidence, 4)},
            )

    # ── Latency + structured inference log ───────────────────────────────────────
    original_ts = original_data.get("timestamp")
    e2e_lat = (time.time() - float(original_ts)) * 1000 if original_ts else 0

    top_3 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    mlog.info(
        "meta_inference",
        extra={
            "event":          "meta_inference",
            "decision_mode":  decision_mode,
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

    # ── Sarcasm / conflict reasoning payload ─────────────────────────────────────
    reasoning = None
    if conflict_desc or sarcasm_score > 0.3:
        reasoning = {
            "type":          "Contextual Dissonance" if conflict_desc else "Sarcastic Intent",
            "details":       conflict_desc or "Emotional signal flip detected.",
            "sarcasm_score": float(sarcasm_score),
            "action":        "Meta-Learner nuance override applied.",
        }

    # ── Logic map (per-block feature impact) ─────────────────────────────────────
    logic_map = calculate_feature_impacts(meta_learner, fv, dominant_emotion) if meta_learner else {}

    # ── Trajectory LSTM step (updates conversation hidden state in Redis) ────────
    trajectory = await run_trajectory_step(
        model=trajectory_model,
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

    # ── Assemble + publish ───────────────────────────────────────────────────────
    pipeline_log = {
        "models":            model_outputs,
        "aggregated":        final_scores,
        "dominant_selected": dominant_emotion,
        "decision_mode":     decision_mode,
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

    # Inject VADER scores into the final payload for the Aggregation/Frontend service.
    if "vader" in model_outputs:
        for k, v in model_outputs["vader"].items():
            final_scores[k] = v

    output_event = original_data.copy()
    output_event["emotions"]         = json.dumps(final_scores)
    output_event["dominant_emotion"] = dominant_emotion
    output_event["pipeline_log"]     = json.dumps(pipeline_log)
    if reasoning:
        output_event["reasoning"] = json.dumps(reasoning)

    # Context divergence detection.
    prev_mood = context.get("prev_emotion", "neutral")
    if prev_mood != "neutral" and dominant_emotion != prev_mood:
        output_event["context_shift"] = json.dumps({
            "type":         "Context Shift",
            "from":         prev_mood,
            "to":           dominant_emotion,
            "significance": "High" if meta_confidence > 0.7 else "Moderate",
        })

    await redis_client.publish_event(OUTPUT_STREAM, output_event)

    METRICS.messages_processed_total.inc()
    METRICS.aggregation_latency_ms.observe(agg_lat)

    # Cleanup
    await r.delete(f"{PENDING_KEY_PREFIX}{message_id}")
