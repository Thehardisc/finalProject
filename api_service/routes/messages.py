"""
api_service/routes/messages.py — Message feedback endpoint.
"""
import time

from fastapi import APIRouter, HTTPException, Depends

from shared.utils.logger import get_logger
from shared.constants import EMOTION_LABELS
from api_service.auth_utils import get_current_user

logger = get_logger("api_service")
router = APIRouter()

_redis_client = None

def set_redis(client):
    global _redis_client
    _redis_client = client


@router.post("/message/{message_id}/feedback",
             dependencies=[Depends(get_current_user)])
async def post_message_feedback(message_id: str, payload: dict):
    label = payload.get("label")
    if not label or label not in EMOTION_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid emotion label. Must be one of: {EMOTION_LABELS}"
        )
    event = {
        "message_id":           message_id,
        "ground_truth_emotion": label,
        "timestamp":            time.time()
    }
    try:
        await _redis_client.publish_event("feedback_stream", event)
        logger.info(f"Feedback received for {message_id}: {label}")
        return {"status": "accepted", "message_id": message_id}
    except Exception as e:
        logger.error(f"Failed to publish feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
