import os
import sys
import time
import traceback

_service_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _service_dir not in sys.path:
    sys.path.insert(0, _service_dir)

from shared.utils.logger import get_logger
from trainer.cycle import run_one_cycle, RETRAIN_INTERVAL, ACCURACY_GATE, MAX_EMPATHETIC_SAMPLES, MIN_DB_SAMPLES, MODEL_PATH, RELOAD_CHANNEL

logger = get_logger("trainer")

if __name__ == "__main__":
    logger.info("Trainer running in standalone container mode (reload via Redis pub/sub).")
    logger.log_stats("Trainer Configuration", {
        "RETRAIN_INTERVAL_SECONDS": RETRAIN_INTERVAL,
        "ACCURACY_GATE":            ACCURACY_GATE,
        "MAX_EMPATHETIC_SAMPLES":   MAX_EMPATHETIC_SAMPLES,
        "MIN_DB_SAMPLES":           MIN_DB_SAMPLES,
        "MODEL_PATH":               str(MODEL_PATH),
        "RELOAD_CHANNEL":           RELOAD_CHANNEL,
    })
    while True:
        try:
            run_one_cycle(reload_callback=None)
        except Exception as e:
            logger.error(f"Standalone trainer cycle failed: {e}")
            traceback.print_exc()
        logger.info(f"Sleeping {RETRAIN_INTERVAL}s until next cycle...")
        time.sleep(RETRAIN_INTERVAL)
