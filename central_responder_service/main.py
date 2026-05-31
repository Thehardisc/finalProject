import asyncio
import sys
import os
import json
import time
from collections import deque
from typing import Optional

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

from trajectory.inference import load_trajectory_model, run_trajectory_step

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from shared.constants import (
    CONTEXT_DIM, N_CDM_STATES,
    CTX_HIST_VALENCE, CTX_RESONANCE, CTX_VOLATILITY, CTX_CURR_VALENCE,
    CTX_RESIDENCY, CTX_ABRUPTNESS,
)
from shared.module_registry import ModuleRegistry

logger = get_logger("central_responder")

redis_client = RedisClient()
INPUT_STREAM  = "partial_analysis_stream"
GROUP_NAME    = "central_responder_group"
CONSUMER_NAME = "responder_1"
OUTPUT_STREAM = "emotion_stream"

AGGREGATION_TIMEOUT_MS = 5000
# How long to wait for optional modules after all required have arrived.
# Tunable via env: set higher (e.g. 100 ms) if context_engine latency is >50 ms p50.
OPTIONAL_TIMEOUT_MS = float(os.environ.get("OPTIONAL_TIMEOUT_MS", "1000"))

# ── Module registry — populated in main() after Redis connects ──────────────────
REGISTRY: Optional[ModuleRegistry] = None

# ── Meta-learner ────────────────────────────────────────────────────────────────
META_LEARNER = load_meta_learner()
if META_LEARNER is None:
    logger.warning("No compatible meta_weights.pkl found. Falling back to Rule-Based Aggregation until trainer finishes.")
else:
    logger.info("Running in META-LEARNER mode.")

def on_model_reload(new_model):
    global META_LEARNER
    META_LEARNER = new_model
    logger.info("Meta-learner hot-reloaded in memory.")

ready_marker = os.path.join(os.path.dirname(__file__), "..", "models", ".ready")
if META_LEARNER is not None:
    open(ready_marker, 'w').close()
    logger.info("✅ [GATEKEEPER] Valid model found. Gate is OPEN.")
else:
    if os.path.exists(ready_marker):
        try:
            os.remove(ready_marker)
        except Exception:
            pass
    logger.info("No valid model found. Missing ready file.")

start_trainer_thread(on_model_reload)
logger.info("Background meta-learner retraining daemon is ACTIVATED.")

_model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
TRAJECTORY_MODEL = load_trajectory_model(
    model_path=os.path.join(_model_dir, 'trajectory_lstm.pt'),
    config_path=os.path.join(_model_dir, 'trajectory_config.json'),
)
if TRAJECTORY_MODEL is not None:
    logger.info("Trajectory LSTM loaded — conversation direction prediction ACTIVE.")


# ── Aggregation helpers ─────────────────────────────────────────────────────────

# Recently-completed message IDs — prevents re-processing late optional arrivals.
# deque(maxlen) automatically evicts from the left when full.
_completed_order: deque = deque(maxlen=500)
_completed_ids:   set   = set()


def _mark_completed(msg_id: str) -> None:
    if len(_completed_ids) >= 500:
        oldest = _completed_order.popleft()
        _completed_ids.discard(oldest)
    _completed_ids.add(msg_id)
    _completed_order.append(msg_id)


async def _do_aggregation(msg_id: str, info: dict, r) -> None:
    """Execute aggregation and clean up pending state."""
    packets    = list(info["models"].values())
    agg_lat    = (time.time() - info["arrival_timestamp"]) * 1000
    try:
        await aggregate_and_publish(msg_id, packets, r, agg_lat=agg_lat)
    except Exception as e:
        logger.error(f"[AGGREGATION FAILED] {msg_id}: {e}", exc_info=True)
    pending_aggregations.pop(msg_id, None)
    _mark_completed(msg_id)


async def _optional_timeout(msg_id: str, r) -> None:
    """
    Fires OPTIONAL_TIMEOUT_MS after all required modules arrived.
    Triggers aggregation with whatever optional modules have appeared so far.
    If aggregation already ran (optional arrived in time), the 'aggregated'
    flag prevents double-processing.
    """
    await asyncio.sleep(OPTIONAL_TIMEOUT_MS / 1000.0)
    info = pending_aggregations.get(msg_id)
    if info and not info.get("aggregated"):
        info["aggregated"] = True   # claim before any await
        optional_present = set(info["models"].keys()) & (REGISTRY.optional if REGISTRY else set())
        logger.debug(
            f"[OPT TIMEOUT] {msg_id}: firing after {OPTIONAL_TIMEOUT_MS}ms  "
            f"optional_present={optional_present}"
        )
        await _do_aggregation(msg_id, info, r)


async def _registry_refresh_loop(registry: ModuleRegistry) -> None:
    """
    Background task: re-reads module_registry Hash every 30 s.
    The initial 5 s delay gives services time to register on startup before
    the first scheduled refresh.
    """
    await asyncio.sleep(5)
    while True:
        try:
            await registry.refresh()
            registry.log_state(logger)
        except Exception as e:
            logger.warning(f"[Registry] periodic refresh failed: {e}")
        await asyncio.sleep(30)


# ── Core aggregation logic ─────────────────────────────────────────────────────

async def aggregate_and_publish(message_id, partial_results, r, agg_lat=0):
    logger.debug(f"Aggregating results for {message_id}...")

    original_data = partial_results[0].get("original_data", {}) if partial_results else {}

    model_outputs  = {}
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
    _nz2 = sum(1 for v in context_vector if v != 0.0) if context_vector else 0
    logger.debug(
        f"[DIAG-2] CE parse msg={message_id} "
        f"ce_available={ce_available} "
        f"dim={len(context_vector) if context_vector else 0} nonzero={_nz2}"
    )
    if not ce_available:
        received_dim = len(context_vector) if context_vector else 0
        if received_dim > 0:
            logger.warning(
                f"[SCHEMA MISMATCH] context_vector dim={received_dim}, "
                f"expected {CONTEXT_DIM}. Injecting zeros."
            )
        context_vector = [0.0] * CONTEXT_DIM

    conv_id     = original_data.get("conversation_id", "conv-1")
    prev_emotion = "neutral"
    try:
        state = await r.hgetall(f"conversation:{conv_id}")
        if state:
            prev_emotion = state.get("dominant_emotion", "neutral")
    except Exception as e:
        logger.warning(f"Failed to fetch prev_emotion for {conv_id}: {e}")

    fv = build_feature_vector(model_outputs, context_vector=context_vector)
    dominant_emotion, meta_confidence, final_scores, sarcasm_score, conflict_desc, gate_alpha = \
        predict_with_meta_learner(META_LEARNER, fv)

    # apply_context_correction is a no-op stub in v2 (gating network handles this natively)
    corrected_scores, ctx_correction_weight = apply_context_correction(
        final_scores, context_vector, prev_emotion=prev_emotion
    )
    if ctx_correction_weight > 0:
        final_scores     = corrected_scores
        dominant_emotion = max(final_scores, key=final_scores.get)
        meta_confidence  = final_scores[dominant_emotion]

    # ── AI Confidence & Anomaly Validation Gatekeeper ───────────────────────────
    bert_scores = model_outputs.get("basic_bert", {})
    bert_max_prob = max([float(v) for v in bert_scores.values()]) if bert_scores else 0.0

    goe_scores = model_outputs.get("go_emotions", {})
    goe_max_prob = max([float(v) for v in goe_scores.values()]) if goe_scores else 0.0

    is_anomaly = False
    anomaly_reason = []

    if bert_max_prob < 0.35 and "basic_bert" in model_outputs:
        anomaly_reason.append(f"Low BERT Confidence ({bert_max_prob:.2f})")
    if goe_max_prob < 0.20 and "go_emotions" in model_outputs:
        anomaly_reason.append(f"Low GoEmotions Confidence ({goe_max_prob:.2f})")
    if meta_confidence < 0.45:
        anomaly_reason.append(f"Low Meta-Learner Confidence ({meta_confidence:.2f})")
    if conflict_desc:
        anomaly_reason.append(f"Model Divergence: {conflict_desc}")

    if (bert_max_prob < 0.35 and goe_max_prob < 0.20) or (meta_confidence < 0.45 and conflict_desc):
        is_anomaly = True
        logger.warning(f"[ANOMALY DETECTED] {message_id}: {', '.join(anomaly_reason)}")

        # Fallback to the most confident NLP model to preserve information
        original_conflict_desc = conflict_desc
        conflict_desc = "Anomaly Fallback: Dynamic Recovery"
        if goe_max_prob >= bert_max_prob and goe_scores:
            final_scores = goe_scores
        elif bert_scores:
            final_scores = bert_scores
        else:
            final_scores = {"neutral": 1.0}

        dominant_emotion = max(final_scores, key=final_scores.get) if final_scores else "neutral"
        meta_confidence = final_scores.get(dominant_emotion, 0.0)

    original_ts = original_data.get("timestamp")
    e2e_lat = (time.time() - float(original_ts)) * 1000 if original_ts else 0

    _ce_packet = next((p for p in partial_results if p.get("model_name") == "context_engine"), {})
    def _named(key, idx):
        v = _ce_packet.get(key)
        return round(float(v), 4) if v is not None else round(float(context_vector[idx]), 4)

    hist_val   = _named("ctx_hist_valence",    CTX_HIST_VALENCE)
    resonance  = _named("ctx_topic_resonance", CTX_RESONANCE)
    volatility = _named("ctx_volatility",      CTX_VOLATILITY)
    cur_val    = _named("ctx_curr_valence",    CTX_CURR_VALENCE)

    top_3 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    stats = {
        "Message ID":       message_id,
        "Dominant Emotion": f"'{dominant_emotion}' ({meta_confidence:.2%})",
        "Prev Emotion":     prev_emotion,
        "CE Context":       f"cur:{cur_val:.3f} hist:{hist_val:.3f} vol:{volatility:.3f} ce={'yes' if ce_available else 'no'}",
        "Top 3":            ", ".join([f"{e}:{s:.2%}" for e, s in top_3]),
        "E2E Latency":      f"{e2e_lat:.2f}ms",
        "Agg Latency":      f"{agg_lat:.2f}ms",
    }
    if conflict_desc:
        stats["Conflict"] = conflict_desc
    logger.log_stats(f"Meta-Inference: {message_id}", stats)

    reasoning = None
    _reason_desc = original_conflict_desc if is_anomaly else conflict_desc
    if _reason_desc or sarcasm_score > 0.3:
        reasoning = {
            "type":          "Anomaly Recovery" if is_anomaly else ("Contextual Dissonance" if _reason_desc else "Sarcastic Intent"),
            "details":       _reason_desc or "Emotional signal flip detected.",
            "sarcasm_score": float(sarcasm_score),
            "action":        "Meta-Learner nuance override applied.",
        }

    logic_map  = calculate_feature_impacts(META_LEARNER, fv, dominant_emotion)

    logger.debug(
        f"[DIAG-4] pre-log msg={message_id} "
        f"ce_available={ce_available} ctx_sum={sum(context_vector):.4f} "
        f"logic_map={logic_map}"
    )

    trajectory = await run_trajectory_step(
        model=TRAJECTORY_MODEL,
        model_outputs=model_outputs,
        conv_id=conv_id,
        redis=r,
        context_vector=context_vector,
    )

    cdm_probs     = context_vector[0:N_CDM_STATES] if ce_available else [0.0] * N_CDM_STATES
    cdm_state_idx = cdm_probs.index(max(cdm_probs)) if ce_available else None

    pipeline_log = {
        "models":                model_outputs,
        "aggregated":            final_scores,
        "dominant_selected":     dominant_emotion,
        "decision_mode":         "meta-learner",
        "meta_confidence":       meta_confidence,
        "logic_map":             logic_map,
        "ctx_correction_weight": ctx_correction_weight,
        "ctx_weight_pct":        round(float(logic_map.get("Context", 0.0)), 4),
        "gate_weights_alpha":    gate_alpha,  # [vader_w, bert_w, goe_w] or None
        "sarcasm_score":         float(sarcasm_score),
        "conflict":              conflict_desc,
        "trajectory":            trajectory,
        "context_snapshot": {
            "prev_emotion":         prev_emotion,
            "cur_valence":          cur_val,
            "historical_valence":   hist_val,
            "topic_resonance":      resonance,
            "volatility":           volatility,
            "ce_available":         ce_available,
            "cdm_state_probs":      [round(float(p), 4) for p in cdm_probs],
            "cdm_current_state":    cdm_state_idx,
            "cdm_residency":        round(float(context_vector[CTX_RESIDENCY]), 3) if ce_available else 0.0,
            "cdm_entry_abruptness": round(float(context_vector[CTX_ABRUPTNESS]), 3) if ce_available else 0.0,
            "cdm_available":        ce_available,
        },
    }

    vader_scalars = {k: v for k, v in model_outputs.get("vader", {}).items()}

    output_event                     = original_data.copy()
    output_event["emotions"]         = json.dumps({**final_scores, **vader_scalars})
    output_event["dominant_emotion"] = dominant_emotion
    output_event["pipeline_log"]     = json.dumps(pipeline_log)
    if reasoning:
        output_event["reasoning"] = json.dumps(reasoning)

    if prev_emotion != "neutral" and dominant_emotion != prev_emotion:
        output_event["context_shift"] = json.dumps({
            "type":         "Context Shift",
            "from":         prev_emotion,
            "to":           dominant_emotion,
            "significance": "High" if meta_confidence > 0.7 else "Moderate",
        })

    if is_anomaly:
        anomaly_payload = {
            "message_id": message_id,
            "original_text": original_data.get("original_text", ""),
            "conversation_id": conv_id,
            "anomaly_reason": ", ".join(anomaly_reason),
            "raw_model_outputs": json.dumps(model_outputs),
            "timestamp": str(time.time())
        }
        await redis_client.publish_event("ai_anomaly_stream", anomaly_payload)

    await redis_client.publish_event(OUTPUT_STREAM, output_event)


# ── Pending aggregations dict ──────────────────────────────────────────────────
# Structure per msg_id:
#   arrival_timestamp : float     — when the first partial result arrived
#   models            : dict      — model_name → full_packet
#   aggregated        : bool      — True once aggregation has been claimed
#   optional_task     : Task|None — the pending _optional_timeout coroutine
pending_aggregations: dict = {}


async def main():
    global REGISTRY

    await redis_client.connect()
    r = redis_client.redis

    # ── Build and seed the module registry ────────────────────────────────────
    REGISTRY = ModuleRegistry(r, refresh_interval=30.0)
    await REGISTRY.refresh()
    REGISTRY.log_state(logger)
    asyncio.create_task(_registry_refresh_loop(REGISTRY))

    try:
        await r.xgroup_create(INPUT_STREAM, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    logger.info(
        f"Central Responder started.  "
        f"OPTIONAL_TIMEOUT_MS={OPTIONAL_TIMEOUT_MS}"
    )

    while True:
        try:
            # Refresh registry cache if stale (cheap: just a time.time() check)
            await REGISTRY.maybe_refresh()

            # PEL recovery: reclaim stale messages idle >30 s
            try:
                _, stale, _ = await r.xautoclaim(
                    INPUT_STREAM, GROUP_NAME, CONSUMER_NAME,
                    min_idle_time=30_000, start_id='0-0', count=10,
                )
                messages = [(INPUT_STREAM, stale)] if stale else None
            except Exception:
                messages = None

            if not messages:
                messages = await r.xreadgroup(
                    GROUP_NAME, CONSUMER_NAME,
                    {INPUT_STREAM: ">"},
                    count=10, block=2000,
                )

            if messages:
                for stream, msgs in messages:
                    for record_id, data in msgs:
                        msg_id     = data.get("message_id")
                        model_name = data.get("model_name")

                        # ── Ignore late arrivals for already-completed messages ──
                        if msg_id in _completed_ids:
                            await r.xack(INPUT_STREAM, GROUP_NAME, record_id)
                            continue

                        # ── Parse the incoming packet ──────────────────────────
                        if model_name == "context_engine":
                            full_packet = {
                                "model_name":          model_name,
                                "context_vector":      data.get("context_vector"),
                                "original_data":       data.get("original_data"),
                                "ctx_hist_valence":    data.get("ctx_hist_valence"),
                                "ctx_topic_resonance": data.get("ctx_topic_resonance"),
                                "ctx_volatility":      data.get("ctx_volatility"),
                                "ctx_curr_valence":    data.get("ctx_curr_valence"),
                                "ctx_dim":             data.get("ctx_dim"),
                            }
                        else:
                            scores_raw = data.get("scores")
                            scores     = scores_raw
                            if isinstance(scores_raw, str):
                                try:
                                    scores = json.loads(scores_raw)
                                except (json.JSONDecodeError, TypeError) as e:
                                    logger.warning(
                                        f"Cannot parse scores for '{model_name}' "
                                        f"(msg {msg_id}): {e}"
                                    )
                            full_packet = {
                                "model_name":    model_name,
                                "scores":        scores,
                                "original_data": data.get("original_data"),
                            }

                        if isinstance(full_packet["original_data"], str):
                            try:
                                full_packet["original_data"] = json.loads(
                                    full_packet["original_data"]
                                )
                            except Exception:
                                pass

                        # ── Accumulate ────────────────────────────────────────
                        if msg_id not in pending_aggregations:
                            pending_aggregations[msg_id] = {
                                "arrival_timestamp": time.time(),
                                "models":            {},
                                "aggregated":        False,
                                "optional_task":     None,
                            }

                        pending_aggregations[msg_id]["models"][model_name] = full_packet

                        # ── Publish partial result for frontend stream progress ─
                        if model_name != "context_engine":
                            orig    = full_packet.get("original_data")
                            conv_id = orig.get("conversation_id") if isinstance(orig, dict) else None
                            if conv_id:
                                try:
                                    await redis_client.publish_event(
                                        "partial_result_stream",
                                        {
                                            "message_id":      msg_id,
                                            "conversation_id": conv_id,
                                            "model":           model_name,
                                        },
                                    )
                                except Exception:
                                    pass

                        # ── Registry-driven aggregation trigger ────────────────
                        info            = pending_aggregations.get(msg_id)
                        if not info or info["aggregated"]:
                            await r.xack(INPUT_STREAM, GROUP_NAME, record_id)
                            continue

                        received_models = set(info["models"].keys())

                        if not REGISTRY.all_required_present(received_models):
                            logger.debug(
                                f"[{msg_id}] waiting for: "
                                f"{REGISTRY.missing_required(received_models)}"
                            )
                            await r.xack(INPUT_STREAM, GROUP_NAME, record_id)
                            continue

                        # All required models present.
                        if REGISTRY.all_optional_present(received_models):
                            # Everything arrived — cancel timeout if scheduled, aggregate now.
                            task = info["optional_task"]
                            if task and not task.done():
                                task.cancel()
                            info["aggregated"] = True
                            logger.debug(
                                f"[{msg_id}] all required + optional present — "
                                f"aggregating immediately"
                            )
                            await _do_aggregation(msg_id, info, r)
                        elif info["optional_task"] is None:
                            # First time required is complete — start the optional timer.
                            logger.debug(
                                f"[{msg_id}] required complete, optional outstanding — "
                                f"starting {OPTIONAL_TIMEOUT_MS}ms timer"
                            )
                            info["optional_task"] = asyncio.create_task(
                                _optional_timeout(msg_id, r)
                            )
                        # else: timer already running, nothing to do this pass.

                        await r.xack(INPUT_STREAM, GROUP_NAME, record_id)

            # ── Stale cleanup (>60 s old, never completed) ─────────────────────
            now        = time.time()
            stale_keys = [
                mid for mid, info in pending_aggregations.items()
                if now - info["arrival_timestamp"] > 60
            ]
            for stale_key in stale_keys:
                info = pending_aggregations.pop(stale_key, {})
                task = info.get("optional_task")
                if task and not task.done():
                    task.cancel()
                logger.warning(
                    f"[STALE] Evicted {stale_key} after 60 s "
                    f"(received: {set(info.get('models', {}).keys())})"
                )

        except Exception as e:
            logger.log_exception("CENTRAL RESPONDER CRITICAL ERROR", e)
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
