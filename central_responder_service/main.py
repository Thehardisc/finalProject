"""
central_responder_service/main.py — Entry point.

Responsibilities:
  1. Load the trained meta-learner model (or warn if missing).
  2. Write / remove the .ready gate file.
  3. Start the background trainer thread.
  4. Run the async Redis consumer loop.
"""
import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger

from central_responder_service.ml.loader      import load_meta_learner
from central_responder_service.trainer.runner  import start_trainer_thread
from central_responder_service.core.stream_consumer import run_consumer_loop

logger = get_logger("central_responder")

redis_client = RedisClient()

# ── Model state (hot-reloadable) ──────────────────────────────────────────────
META_LEARNER = load_meta_learner()
if META_LEARNER is None:
    logger.warning(
        "No compatible meta_weights.pkl found. "
        "Falling back to Rule-Based Aggregation until trainer finishes."
    )
else:
    logger.info("Running in META-LEARNER mode.")

# ── Ready gate ────────────────────────────────────────────────────────────────
READY_MARKER = os.path.join(os.path.dirname(__file__), "..", "models", ".ready")

if META_LEARNER is not None:
    open(READY_MARKER, 'w').close()
    logger.info("✅ [GATEKEEPER] Valid model found. Gate is OPEN.")
else:
    if os.path.exists(READY_MARKER):
        try:
            os.remove(READY_MARKER)
        except Exception:
            pass
    logger.info("No valid model found. Gate is CLOSED until training completes.")


# ── Hot-reload callback ───────────────────────────────────────────────────────
def on_model_reload(new_model):
    global META_LEARNER
    META_LEARNER = new_model
    logger.info("Meta-learner hot-reloaded in memory.")


def get_meta_learner():
    return META_LEARNER


# ── Start background trainer ──────────────────────────────────────────────────
start_trainer_thread(on_model_reload)
logger.info("Background meta-learner retraining daemon ACTIVATED.")


# ── Async entry point ─────────────────────────────────────────────────────────
async def main():
    await redis_client.connect()
    await run_consumer_loop(redis_client, get_meta_learner)


if __name__ == "__main__":
    asyncio.run(main())
