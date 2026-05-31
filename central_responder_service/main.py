"""
main.py — Central Responder entrypoint.

Thin bootstrap: load the meta-learner, wire hot-reload, start the background
trainer + Prometheus metrics, and hand the Redis stream loop to
core.stream_consumer. All fusion/inference logic lives in core/ + ml/.
"""
import asyncio
import sys
import os
import json

# Add parent directory to path to import shared (must come before shared imports).
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.loader import load_meta_learner
from trainer import start_trainer_thread
from trajectory.inference import load_trajectory_model
from core.stream_consumer import run_consumer_loop

import metrics as METRICS  # noqa: F401  (kept so the module is imported once at startup)
from prometheus_client import start_http_server as _start_metrics_server

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger

logger = get_logger("central_responder")

redis_client = RedisClient()

_MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
_READY_MARKER = os.path.join(_MODEL_DIR, ".ready")
_META_JSON    = os.path.join(_MODEL_DIR, "meta_weights_meta.json")

# Single mutable holder so the consumer's get_meta_learner() callable always reads
# the current model — hot-reload swaps it in place without restarting the loop.
_META = {"model": load_meta_learner()}


def get_meta_learner():
    return _META["model"]


def on_model_reload(new_model):
    """Hot-reload callback invoked by the trainer thread after a successful deploy."""
    prev_acc = new_acc = improvement = None
    try:
        if os.path.exists(_META_JSON):
            with open(_META_JSON) as f:
                meta = json.load(f)
            new_acc     = meta.get("test_accuracy")
            prev_acc    = meta.get("previous_accuracy")
            improvement = meta.get("improvement")
    except Exception as e:
        logger.warning(
            f"hot_reload_meta_read_failed error={type(e).__name__}",
            extra={"extra_data": {"event": "hot_reload_meta_read_failed", "error": str(e)}},
        )

    _META["model"] = new_model
    logger.info(
        f"model_hot_reload prev_acc={prev_acc} new_acc={new_acc} improvement={improvement}",
        extra={"extra_data": {
            "event":       "model_hot_reload",
            "prev_acc":    prev_acc,
            "new_acc":     new_acc,
            "improvement": improvement,
        }},
    )


def _init_gatekeeper():
    """Write/remove the .ready marker the api_service health check reads."""
    if _META["model"] is not None:
        open(_READY_MARKER, "w").close()
        logger.info("Running in META-LEARNER mode. [GATEKEEPER] Gate is OPEN.")
    else:
        if os.path.exists(_READY_MARKER):
            try:
                os.remove(_READY_MARKER)
            except OSError:
                pass
        logger.warning(
            "No compatible meta_weights.pkl found. Falling back to RULE-BASED "
            "aggregation until the trainer finishes. [GATEKEEPER] Gate is CLOSED."
        )


async def main():
    # Start the Prometheus /metrics server early so it's scrapeable during slow startup.
    try:
        _start_metrics_server(9090)
        logger.info("Prometheus metrics server listening on :9090/metrics")
    except OSError as e:
        logger.warning(f"Could not start metrics server on :9090: {e}")

    _init_gatekeeper()
    start_trainer_thread(on_model_reload)
    logger.info("Background meta-learner retraining daemon ACTIVATED.")

    trajectory_model = load_trajectory_model(
        model_path=os.path.join(_MODEL_DIR, "trajectory_lstm.pt"),
        config_path=os.path.join(_MODEL_DIR, "trajectory_config.json"),
    )
    if trajectory_model is not None:
        logger.info("Trajectory LSTM loaded — conversation direction prediction ACTIVE.")

    await redis_client.connect()
    await run_consumer_loop(redis_client, get_meta_learner, trajectory_model)


if __name__ == "__main__":
    asyncio.run(main())
