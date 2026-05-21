"""
api_service/routes/messages.py — Message feedback and delete endpoints.
"""
import time

from fastapi import APIRouter, HTTPException, Depends

from shared.utils.logger import get_logger
from shared.constants import EMOTION_LABELS
from api_service.auth_utils import get_current_user
from api_service.db.pool import get_pool

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


@router.delete("/message/{message_id}")
async def delete_message(message_id: str,
                          current_user: dict = Depends(get_current_user)):
    """Delete a message. Only the sender can delete their own message."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id FROM messages WHERE message_id = $1", message_id
        )
        if not row:
            raise HTTPException(status_code=404, detail="Message not found.")
        if row["user_id"] != current_user["sub"]:
            raise HTTPException(status_code=403,
                                detail="You can only delete your own messages.")

        async with conn.transaction():
            await conn.execute(
                "DELETE FROM emotion_analysis WHERE message_id = $1", message_id
            )
            await conn.execute(
                "DELETE FROM messages WHERE message_id = $1", message_id
            )

    logger.info(f"Message {message_id} deleted by {current_user['sub']}")
    return {"status": "deleted", "message_id": message_id}
