"""
core/aggregator.py — Collect partial model results and publish the final prediction.
"""
import json
import time

from shared.utils.logger import get_logger
from shared.utils.redis_client import RedisClient

from ..ml.predictor       import build_feature_vector, predict_with_meta_learner
from ..ml.impact_calculator import calculate_feature_impacts

logger = get_logger("central_responder")

# All 27 GoEmotions labels (used for stream validation)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

OUTPUT_STREAM = "emotion_stream"


async def aggregate_and_publish(message_id: str, partial_results: list,
                                 meta_learner, redis_client: RedisClient,
                                 r, agg_lat: float = 0) -> None:
    """
    Build the feature vector from all 4 model results, run the meta-learner,
    and publish the enriched result to the emotion_stream.
    """
    logger.debug(f"Aggregating results for {message_id}...")

    original_data = partial_results[0].get("original_data", {}) if partial_results else {}

    # Collect model outputs
    model_outputs = {}
    for res in partial_results:
        model_name = res.get("model_name")
        scores     = res.get("scores", {})
        model_outputs[model_name] = scores

    # Fetch conversation context from Redis
    conv_id = original_data.get("conversation_id", "conv-1")
    context = {"avg_valence": 0.0, "prev_emotion": "neutral"}
    try:
        state_key = f"conversation:{conv_id}"
        state = await r.hgetall(state_key)
        if state:
            context["avg_valence"]  = float(state.get("average_valence", 0.0))
            context["prev_emotion"] = state.get("dominant_emotion", "neutral")
            logger.debug(f"Context injected for {conv_id}: "
                         f"Valence={context['avg_valence']}, Mood={context['prev_emotion']}")
    except Exception as e:
        logger.warning(f"Failed to fetch context for {conv_id}: {e}")

    # Run inference
    fv = build_feature_vector(model_outputs, context=context)
    dominant_emotion, meta_confidence, final_scores, sarcasm_score, conflict_desc = \
        predict_with_meta_learner(meta_learner, fv)

    # Latency
    original_ts = original_data.get("timestamp")
    e2e_lat = (time.time() - float(original_ts)) * 1000 if original_ts else 0

    # Log stats block
    top_3 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    stats = {
        "Message ID":         message_id,
        "Dominant Emotion":   f"'{dominant_emotion}' ({meta_confidence:.2%})",
        "Context Injected":   f"Val:{context['avg_valence']:.2f}, Mood:{context['prev_emotion']}",
        "Top 3 Probabilities": ", ".join([f"{e}:{s:.2%}" for e, s in top_3]),
        "E2E Latency":        f"{e2e_lat:.2f}ms",
        "Agg Latency":        f"{agg_lat:.2f}ms"
    }
    if conflict_desc:
        stats["Conflict"] = conflict_desc
    logger.log_stats(f"Meta-Inference: {message_id}", stats)

    # Sarcasm reasoning payload
    reasoning = None
    if conflict_desc or sarcasm_score > 0.3:
        reasoning = {
            "type":         "Contextual Dissonance" if conflict_desc else "Sarcastic Intent",
            "details":      conflict_desc or "Emotional signal flip detected.",
            "sarcasm_score": float(sarcasm_score),
            "action":       "Meta-Learner nuance override applied."
        }

    # Logic map
    logic_map = calculate_feature_impacts(meta_learner, fv, dominant_emotion)

    pipeline_log = {
        "models":           model_outputs,
        "aggregated":       final_scores,
        "dominant_selected": dominant_emotion,
        "decision_mode":    "meta-learner",
        "meta_confidence":  meta_confidence,
        "logic_map":        logic_map,
        "sarcasm_score":    float(sarcasm_score),
        "conflict":         conflict_desc
    }

    # Inject VADER scores into final payload for frontend
    if "vader" in model_outputs:
        for k, v in model_outputs["vader"].items():
            final_scores[k] = v

    output_event = original_data.copy()
    output_event["emotions"]      = json.dumps(final_scores)
    output_event["dominant_emotion"] = dominant_emotion
    output_event["pipeline_log"]  = json.dumps(pipeline_log)
    if reasoning:
        output_event["reasoning"] = json.dumps(reasoning)

    # Context divergence detection
    prev_mood = context.get("prev_emotion", "neutral")
    if prev_mood != "neutral" and dominant_emotion != prev_mood:
        divergence = {
            "type":        "Context Shift",
            "from":        prev_mood,
            "to":          dominant_emotion,
            "significance": "High" if meta_confidence > 0.7 else "Moderate"
        }
        output_event["context_shift"] = json.dumps(divergence)

    await redis_client.publish_event(OUTPUT_STREAM, output_event)
    await r.delete(f"pending_aggregation:{message_id}")
