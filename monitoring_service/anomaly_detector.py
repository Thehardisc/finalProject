import asyncio
import json
import os
import signal
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger

logger = get_logger("anomaly_detector")

redis_client  = RedisClient()
STREAM_KEY    = "ai_anomaly_stream"
GROUP_NAME    = "anomaly_monitor_group"
CONSUMER_NAME = "monitor_1"
DATASET_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "anomalies.jsonl")

_shutdown = False

async def main():
    await redis_client.connect()
    r = redis_client.redis

    loop = asyncio.get_event_loop()
    def _handle_sigterm():
        global _shutdown
        logger.info("[anomaly_detector] SIGTERM received. Shutting down gracefully.")
        _shutdown = True
    loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)

    try:
        await r.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    logger.info(f"Anomaly Detector started. Writing to {DATASET_PATH}")
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    # Track anomalies in memory to alert on spikes (Drift Detection)
    recent_anomalies = []

    while True:
        try:
            messages = await r.xreadgroup(
                GROUP_NAME, CONSUMER_NAME,
                {STREAM_KEY: ">"}, count=10, block=2000
            )

            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        try:
                            # 1. Log the Anomaly
                            orig_text = data.get("original_text", "")
                            reason = data.get("anomaly_reason", "Unknown")
                            logger.error(f"[AI DRIFT] Detected Anomaly: {reason} | Text: '{orig_text}'")

                            # 2. Persist to Dataset for Meta-Learner Retraining
                            with open(DATASET_PATH, "a") as f:
                                f.write(json.dumps(data) + "\n")
                                
                            # 3. Track frequency
                            recent_anomalies.append(time.time())

                            # ACK the message
                            await r.xack(STREAM_KEY, GROUP_NAME, message_id)

                        except Exception as msg_err:
                            logger.error(f"Failed to process anomaly record {message_id}: {msg_err}")
                            try:
                                await r.xack(STREAM_KEY, GROUP_NAME, message_id)
                            except Exception:
                                pass
                                
            # Drift Detection Logic (e.g., more than 5 anomalies in the last 60 seconds)
            current_time = time.time()
            recent_anomalies = [t for t in recent_anomalies if current_time - t < 60]
            if len(recent_anomalies) >= 5:
                logger.critical(f"[DRIFT ALERT] Severe Model Divergence! {len(recent_anomalies)} anomalies in the last minute.")
                recent_anomalies.clear() # Reset after alert

        except Exception as e:
            logger.log_exception("ANOMALY DETECTOR CRITICAL ERROR", e)
            await asyncio.sleep(2)

        if _shutdown:
            logger.info("Anomaly Detector graceful shutdown complete.")
            break

if __name__ == "__main__":
    asyncio.run(main())
