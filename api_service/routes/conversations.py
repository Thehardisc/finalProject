"""
api_service/routes/conversations.py — Conversation and message retrieval endpoints.
"""
import time
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from shared.utils.logger import get_logger
from api_service.db.pool import get_pool
from api_service.auth_utils import get_current_user

logger = get_logger("api_service")
router = APIRouter()

_redis_client = None

def set_redis(client):
    global _redis_client
    _redis_client = client


class CreateConversationRequest(BaseModel):
    user_id:        str
    target_user_id: str


class CreateGroupRequest(BaseModel):
    name:       str = Field(..., min_length=1, max_length=100)
    member_ids: List[str] = Field(..., min_items=1, max_items=19)


class AddMemberRequest(BaseModel):
    user_id: str


@router.post("/conversations")
async def create_conversation(req: CreateConversationRequest):
    if req.user_id == req.target_user_id:
        raise HTTPException(status_code=400, detail="Cannot create a conversation with yourself.")
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


@router.post("/conversations/group", dependencies=[Depends(get_current_user)])
async def create_group_conversation(req: CreateGroupRequest,
                                     current_user: dict = Depends(get_current_user)):
    creator_id = current_user["sub"]
    all_member_ids = list(set(req.member_ids))
    if creator_id in all_member_ids:
        all_member_ids.remove(creator_id)

    pool = get_pool()
    async with pool.acquire() as conn:
        # Validate all provided user IDs exist
        for uid in all_member_ids:
            exists = await conn.fetchval(
                "SELECT 1 FROM users WHERE user_id = $1 AND is_active = TRUE", uid
            )
            if not exists:
                raise HTTPException(status_code=400, detail=f"User {uid} not found.")

        conv_id = f"grp-{str(uuid.uuid4())[:8]}"
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO conversations (conversation_id, type, name, creator_user_id, created_at) "
                "VALUES ($1, 'group', $2, $3, $4)",
                conv_id, req.name.strip(), creator_id, time.time()
            )
            for uid in [creator_id] + all_member_ids:
                await conn.execute(
                    "INSERT INTO conversation_participants "
                    "(conversation_id, user_id, joined_at) VALUES ($1, $2, $3)",
                    conv_id, uid, time.time()
                )
    logger.info(f"Group '{req.name}' created by {creator_id} with {len(all_member_ids)+1} members")
    return {"conversation_id": conv_id, "name": req.name.strip(), "type": "group"}


@router.post("/conversations/{conv_id}/members")
async def add_group_member(conv_id: str, req: AddMemberRequest,
                            current_user: dict = Depends(get_current_user)):
    pool = get_pool()
    async with pool.acquire() as conn:
        conv = await conn.fetchrow(
            "SELECT type, creator_user_id FROM conversations WHERE conversation_id = $1",
            conv_id
        )
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        if conv["type"] != "group":
            raise HTTPException(status_code=400, detail="Only group conversations support member management.")
        if conv["creator_user_id"] != current_user["sub"]:
            raise HTTPException(status_code=403, detail="Only the group creator can add members.")

        already = await conn.fetchval(
            "SELECT 1 FROM conversation_participants WHERE conversation_id = $1 AND user_id = $2",
            conv_id, req.user_id
        )
        if already:
            raise HTTPException(status_code=409, detail="User is already a member.")

        user_exists = await conn.fetchval(
            "SELECT 1 FROM users WHERE user_id = $1 AND is_active = TRUE", req.user_id
        )
        if not user_exists:
            raise HTTPException(status_code=404, detail="User not found.")

        await conn.execute(
            "INSERT INTO conversation_participants (conversation_id, user_id, joined_at) "
            "VALUES ($1, $2, $3)",
            conv_id, req.user_id, time.time()
        )
    return {"status": "added", "user_id": req.user_id}


@router.delete("/conversations/{conv_id}/members/{target_user_id}")
async def remove_group_member(conv_id: str, target_user_id: str,
                               current_user: dict = Depends(get_current_user)):
    pool = get_pool()
    async with pool.acquire() as conn:
        conv = await conn.fetchrow(
            "SELECT type, creator_user_id FROM conversations WHERE conversation_id = $1",
            conv_id
        )
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        if conv["type"] != "group":
            raise HTTPException(status_code=400, detail="Only group conversations support member management.")
        if conv["creator_user_id"] != current_user["sub"]:
            raise HTTPException(status_code=403, detail="Only the group creator can remove members.")
        if target_user_id == current_user["sub"]:
            raise HTTPException(status_code=400, detail="Creator cannot remove themselves.")

        is_member = await conn.fetchval(
            "SELECT 1 FROM conversation_participants WHERE conversation_id = $1 AND user_id = $2",
            conv_id, target_user_id
        )
        if not is_member:
            raise HTTPException(status_code=404, detail="User is not a member of this group.")

        await conn.execute(
            "DELETE FROM conversation_participants WHERE conversation_id = $1 AND user_id = $2",
            conv_id, target_user_id
        )
    return {"status": "removed", "user_id": target_user_id}


@router.get("/conversations/{user_id}")
async def my_conversations(user_id: str):
    pool = get_pool()
    async with pool.acquire() as conn:
        # Direct conversations
        direct_rows = await conn.fetch(
            """
            SELECT c.conversation_id, c.type, c.created_at,
                   u.display_name AS other_display_name,
                   u.user_id      AS other_user_id,
                   NULL::varchar  AS name,
                   NULL::varchar  AS creator_user_id
            FROM conversations c
            JOIN conversation_participants my_p    ON my_p.conversation_id    = c.conversation_id
            JOIN conversation_participants other_p ON other_p.conversation_id = c.conversation_id
                                                   AND other_p.user_id != $1
            JOIN users u ON other_p.user_id = u.user_id
            WHERE my_p.user_id = $1 AND c.type = 'direct'
              AND u.user_id != $1
            ORDER BY c.created_at DESC
            """,
            user_id
        )

        # Group conversations
        group_rows = await conn.fetch(
            """
            SELECT c.conversation_id, c.type, c.created_at,
                   NULL::varchar AS other_display_name,
                   NULL::varchar AS other_user_id,
                   c.name,
                   c.creator_user_id
            FROM conversations c
            JOIN conversation_participants p ON p.conversation_id = c.conversation_id
            WHERE p.user_id = $1 AND c.type = 'group'
            ORDER BY c.created_at DESC
            """,
            user_id
        )

        # Fetch member lists for all groups in one query
        group_members = {}
        if group_rows:
            group_ids = [row["conversation_id"] for row in group_rows]
            all_members = await conn.fetch(
                """
                SELECT p.conversation_id, u.user_id, u.display_name
                FROM conversation_participants p
                JOIN users u ON u.user_id = p.user_id
                WHERE p.conversation_id = ANY($1)
                """,
                group_ids
            )
            for m in all_members:
                cid = m["conversation_id"]
                group_members.setdefault(cid, []).append(
                    {"user_id": m["user_id"], "display_name": m["display_name"]}
                )

    all_rows = list(direct_rows) + list(group_rows)

    # Fetch all Redis conversation states in one pipeline instead of one-by-one
    redis_states = {}
    if _redis_client and _redis_client.redis and all_rows:
        pipe = _redis_client.redis.pipeline()
        for row in all_rows:
            pipe.hgetall(f"conversation:{row['conversation_id']}")
        states = await pipe.execute()
        for row, state in zip(all_rows, states):
            redis_states[row["conversation_id"]] = state

    result = []
    for row in all_rows:
        d = dict(row)
        cid = d["conversation_id"]

        if d["type"] == "group":
            d["members"] = group_members.get(cid, [])
            d["member_count"] = len(d["members"])

        state = redis_states.get(cid)
        if state:
            d["average_valence"]  = float(state.get("average_valence", 0.0))
            d["dominant_emotion"] = state.get("dominant_emotion", "Neutral")
        else:
            d["average_valence"]  = 0.0
            d["dominant_emotion"] = "Neutral"

        result.append(d)

    result.sort(key=lambda x: x.get("created_at", 0), reverse=True)
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
