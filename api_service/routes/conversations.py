"""
api_service/routes/conversations.py — Conversation and message retrieval endpoints.
"""
import time
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from shared.utils.logger import get_logger
from api_service.db.pool import get_pool
from api_service.auth_utils import get_current_user

logger = get_logger("api_service")
router = APIRouter()

# redis_client injected at app startup via dependency state (see main.py)
_redis_client = None

def set_redis(client):
    global _redis_client
    _redis_client = client


class CreateConversationRequest(BaseModel):
    user_id:        str
    target_user_id: str


@router.post("/conversations")
async def create_conversation(req: CreateConversationRequest):
    pool = get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchval(
            """
            SELECT c.conversation_id
            FROM conversations c
            JOIN conversation_participants p1 ON c.conversation_id = p1.conversation_id
            JOIN conversation_participants p2 ON c.conversation_id = p2.conversation_id
            WHERE c.type = 'direct' AND p1.user_id = $1 AND p2.user_id = $2
            """,
            req.user_id, req.target_user_id
        )
        if existing:
            return {"conversation_id": existing}

        conv_id = f"conv-{str(uuid.uuid4())[:8]}"
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO conversations (conversation_id, type, created_at) "
                "VALUES ($1, 'direct', $2)", conv_id, time.time()
            )
            for uid in [req.user_id, req.target_user_id]:
                await conn.execute(
                    "INSERT INTO conversation_participants "
                    "(conversation_id, user_id, joined_at) VALUES ($1, $2, $3)",
                    conv_id, uid, time.time()
                )
    return {"conversation_id": conv_id}


@router.get("/conversations/{user_id}")
async def my_conversations(user_id: str):
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.conversation_id, c.type, c.created_at,
                   u.display_name AS other_display_name,
                   u.user_id      AS other_user_id
            FROM conversations c
            JOIN conversation_participants my_p    ON my_p.conversation_id    = c.conversation_id
            JOIN conversation_participants other_p ON other_p.conversation_id = c.conversation_id
                                                   AND other_p.user_id != $1
            JOIN users u ON other_p.user_id = u.user_id
            WHERE my_p.user_id = $1
            ORDER BY c.created_at DESC
            """,
            user_id
        )

    result = []
    for row in rows:
        d = dict(row)
        if _redis_client and _redis_client.redis:
            state = await _redis_client.redis.hgetall(
                f"conversation:{d['conversation_id']}"
            )
            d['average_valence']  = float(state.get("average_valence", 0.0)) if state else 0.0
            d['dominant_emotion'] = state.get("dominant_emotion", "Neutral") if state else "Neutral"
        else:
            d['average_valence']  = 0.0
            d['dominant_emotion'] = "Neutral"
        result.append(d)
    return result


@router.get("/conversation/{conversation_id}/state",
            dependencies=[Depends(get_current_user)])
async def get_conversation_state(conversation_id: str):
    if not _redis_client or not _redis_client.redis:
        return {"message_count": 0, "overall_mood": "Neutral",
                "average_valence": 0.0, "conversation_id": conversation_id,
                "status": "New"}
    state = await _redis_client.redis.hgetall(f"conversation:{conversation_id}")
    if not state:
        return {"message_count": 0, "overall_mood": "Neutral",
                "average_valence": 0.0, "conversation_id": conversation_id,
                "status": "New"}
    return state


@router.get("/conversation/{conversation_id}/messages",
            dependencies=[Depends(get_current_user)])
async def get_conversation_messages(conversation_id: str, limit: int = 50):
    """Get messages with their latest emotion analysis for a conversation."""
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT m.message_id AS id, m.text AS content, m.timestamp,
                       m.user_id AS sender_id,
                       a.emotions_json   AS emotions,
                       a.reasoning_json  AS reasoning,
                       a.pipeline_log_json AS pipeline_log
                FROM messages m
                LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
                WHERE m.conversation_id = $1
                ORDER BY m.timestamp DESC
                LIMIT $2
                """,
                conversation_id, limit
            )
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return []
