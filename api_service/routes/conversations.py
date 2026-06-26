"""
api_service/routes/conversations.py — Conversation and message retrieval endpoints.
"""
import json
import time
import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from shared.utils.logger import get_logger
from api_service.db.pool import get_pool
from api_service.auth_utils import get_current_user

logger = get_logger("api_service")
router = APIRouter()

_redis_client = None
_invalidate_participant_cache = None

def set_redis(client):
    global _redis_client
    _redis_client = client

def set_cache_invalidator(fn):
    global _invalidate_participant_cache
    _invalidate_participant_cache = fn


class CreateConversationRequest(BaseModel):
    user_id:        str
    target_user_id: str


class CreateGroupRequest(BaseModel):
    name:       str = Field(..., min_length=1, max_length=100)
    member_ids: List[str] = Field(..., min_items=1, max_items=19)


class AddMemberRequest(BaseModel):
    user_id: str


@router.post("/conversations")
async def create_conversation(req: CreateConversationRequest,
                               current_user: dict = Depends(get_current_user)):
    user_id = current_user["sub"]
    if user_id == req.target_user_id:
        raise HTTPException(status_code=400, detail="Cannot create a conversation with yourself.")
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            existing = await conn.fetchval(
                """
                SELECT c.conversation_id
                FROM conversations c
                JOIN conversation_participants p1 ON c.conversation_id = p1.conversation_id
                JOIN conversation_participants p2 ON c.conversation_id = p2.conversation_id
                WHERE c.type = 'direct' AND p1.user_id = $1 AND p2.user_id = $2
                """,
                user_id, req.target_user_id
            )
            if existing:
                return {"conversation_id": existing}

            conv_id = f"conv-{str(uuid.uuid4())[:8]}"
            await conn.execute(
                "INSERT INTO conversations (conversation_id, type, created_at) "
                "VALUES ($1, 'direct', $2)", conv_id, time.time()
            )
            for uid in [user_id, req.target_user_id]:
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
    if _invalidate_participant_cache:
        _invalidate_participant_cache(conv_id)
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
    if _invalidate_participant_cache:
        _invalidate_participant_cache(conv_id)
    return {"status": "removed", "user_id": target_user_id}


@router.get("/conversations/{user_id}")
async def my_conversations(user_id: str, current_user: dict = Depends(get_current_user)):
    if user_id != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Access denied.")
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


@router.get("/conversation/{conversation_id}/emotional-state",
            dependencies=[Depends(get_current_user)])
async def get_emotional_state(conversation_id: str):
    """
    Bot-readable emotional state for a conversation.

    Returns the narrative mood, emotional trajectory, and recent mood arc
    computed from the full message history — not just the last message.

    Intended for bots and agents that need to adapt their response tone
    based on the conversation's emotional context.

    trajectory values:
      escalating     — conversation is getting more negative
      de-escalating  — conversation is getting more positive
      stable         — no significant change

    arc: list of up to 5 recent mood labels, newest first.
    """
    _EMPTY = {
        "conversation_id":   conversation_id,
        "current_mood":      "neutral",
        "dominant_emotion":  "neutral",
        "trajectory":        "stable",
        "arc":               [],
        "ema_valence":       0.0,
        "message_count":     0,
        "status":            "no_data",
    }

    if not _redis_client or not _redis_client.redis:
        return _EMPTY

    r   = _redis_client.redis
    raw = await r.hgetall(f"conversation:{conversation_id}:emotional_state")

    if not raw:
        return _EMPTY

    arc_raw = raw.get("arc", "[]")
    try:
        import json as _json
        arc = _json.loads(arc_raw)
    except Exception:
        arc = []

    return {
        "conversation_id":   conversation_id,
        "current_mood":      raw.get("current_mood",     "neutral"),
        "dominant_emotion":  raw.get("dominant_emotion", "neutral"),
        "trajectory":        raw.get("trajectory",       "stable"),
        "arc":               arc,
        "ema_valence":       float(raw.get("ema_valence",    0.0)),
        "message_count":     int(raw.get("message_count",   0)),
        "status":            "active",
    }


_EMOTION_TO_MOOD: dict = {
    'admiration': 'warm',      'amusement':      'joyful',
    'approval':   'warm',      'caring':         'warm',
    'curiosity':  'neutral',   'desire':         'warm',
    'excitement': 'joyful',    'gratitude':      'warm',
    'joy':        'joyful',    'love':           'warm',
    'optimism':   'joyful',    'pride':          'joyful',
    'relief':     'resolved',  'realization':    'neutral',
    'surprise':   'neutral',   'neutral':        'neutral',
    'anger':      'hostile',   'annoyance':      'hostile',
    'disapproval':'conflicted','disgust':        'hostile',
    'disappointment': 'melancholic',
    'embarrassment':  'conflicted',
    'fear':       'anxious',   'grief':          'melancholic',
    'nervousness':'anxious',   'remorse':        'melancholic',
    'sadness':    'melancholic','confusion':     'conflicted',
}

_MOOD_VALENCE: dict = {
    'joyful': 0.75, 'warm': 0.55, 'resolved': 0.30, 'neutral': 0.0,
    'anxious': -0.35, 'conflicted': -0.40, 'melancholic': -0.65, 'hostile': -0.80,
}


def _dominant_from_emotions(emotions: dict) -> str:
    skip = {'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'dominant_emotion'}
    best, best_score = 'neutral', 0.0
    for k, v in emotions.items():
        if k not in skip and isinstance(v, (int, float)) and v > best_score:
            best, best_score = k, v
    return best


def _build_chunks(rows: list) -> list:
    """
    Group consecutive messages into emotionally coherent chunks.
    A new chunk starts when the mood label changes.
    Each row: (message_id, text, timestamp, user_id, emotions_json_str)
    """
    chunks, current_chunk, current_mood = [], [], None

    for i, (msg_id, text, ts, uid, emo_json) in enumerate(rows):
        try:
            emotions = json.loads(emo_json) if emo_json else {}
        except Exception:
            emotions = {}

        dominant = _dominant_from_emotions(emotions)
        mood     = _EMOTION_TO_MOOD.get(dominant, 'neutral')

        if mood != current_mood:
            if current_chunk:
                chunks.append((current_mood, current_chunk))
            current_chunk = []
            current_mood  = mood

        current_chunk.append({
            "index":            i,
            "message_id":       msg_id,
            "text":             text,
            "dominant_emotion": dominant,
            "emotions":         {k: float(v) for k, v in emotions.items()
                                 if k not in {'vader_neg', 'vader_neu',
                                              'vader_pos', 'vader_compound',
                                              'dominant_emotion'}},
        })

    if current_chunk:
        chunks.append((current_mood, current_chunk))

    result = []
    for chunk_idx, (mood, msgs) in enumerate(chunks):
        # Average emotion scores across messages in the chunk
        all_keys = set()
        for m in msgs:
            all_keys.update(m["emotions"].keys())
        avg_emotions = {}
        for k in all_keys:
            vals = [m["emotions"].get(k, 0.0) for m in msgs]
            avg_emotions[k] = round(sum(vals) / len(vals), 4)

        result.append({
            "chunk_index":    chunk_idx,
            "message_indices": [m["index"] for m in msgs],
            "message_ids":    [m["message_id"] for m in msgs],
            "mood":           mood,
            "emotions":       avg_emotions,
            "transition_logic": (
                f"Initial state: {mood}." if chunk_idx == 0
                else f"Transition [{result[chunk_idx-1]['mood']} → {mood}]."
            ),
        })

    return result


async def _run_analysis(conversation_id: str, r) -> dict:
    """Fetch messages from DB, build chunks, write results to Redis."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT m.message_id, m.text, m.timestamp, m.user_id,
                   ea.emotions_json
            FROM messages m
            LEFT JOIN emotion_analysis ea ON m.message_id = ea.message_id
            WHERE m.conversation_id = $1
            ORDER BY m.timestamp ASC
            """,
            conversation_id,
        )

    if not rows:
        return {"status": "no_messages"}

    chunks = _build_chunks(list(rows))

    if not chunks:
        return {"status": "no_chunks"}

    # Overall trajectory: compare first chunk mood valence vs last
    first_v = _MOOD_VALENCE.get(chunks[0]["mood"],  0.0)
    last_v  = _MOOD_VALENCE.get(chunks[-1]["mood"], 0.0)
    delta   = last_v - first_v
    trajectory = "escalating" if delta < -0.2 else ("de-escalating" if delta > 0.2 else "stable")

    last_chunk  = chunks[-1]
    current_mood = last_chunk["mood"]
    dominant     = max(last_chunk["emotions"], key=last_chunk["emotions"].get,
                       default="neutral") if last_chunk["emotions"] else "neutral"

    # Write full chunk analysis to Redis
    analysis = {
        "conversation_id": conversation_id,
        "chunks":          json.dumps(chunks),
        "current_mood":    current_mood,
        "dominant_emotion":dominant,
        "trajectory":      trajectory,
        "message_count":   len(rows),
        "chunk_count":     len(chunks),
        "analyzed_at":     time.time(),
        "mode":            "local",
    }

    pipe = r.pipeline()
    # Full analysis key (for training data export + deep inspection)
    pipe.hset(f"conversation:{conversation_id}:analysis",
              mapping={k: str(v) for k, v in analysis.items()})
    pipe.expire(f"conversation:{conversation_id}:analysis", 86400 * 30)
    # Overwrite emotional_state so bot sees the post-conversation view
    pipe.hset(f"conversation:{conversation_id}:emotional_state", mapping={
        "current_mood":     current_mood,
        "dominant_emotion": dominant,
        "trajectory":       trajectory,
        "arc":              json.dumps([c["mood"] for c in chunks[-5:]]),
        "ema_valence":      round(last_v, 4),
        "message_count":    len(rows),
        "last_updated":     time.time(),
    })
    pipe.expire(f"conversation:{conversation_id}:emotional_state", 86400 * 30)
    await pipe.execute()

    logger.info(
        "conversation_analyzed",
        extra={
            "event":           "conversation_analyzed",
            "conversation_id": conversation_id,
            "chunks":          len(chunks),
            "trajectory":      trajectory,
            "mode":            "local",
        },
    )
    return analysis


@router.post("/conversation/{conversation_id}/analyze",
             dependencies=[Depends(get_current_user)])
async def analyze_conversation(conversation_id: str, background_tasks: BackgroundTasks):
    """
    Trigger post-conversation emotional analysis.

    Fetches all messages from DB, groups them into emotionally coherent
    chunks, and writes the results to Redis so the bot can read the
    full narrative emotional arc — not just the last message.

    The analysis runs as a background task (returns immediately).
    Poll GET /conversation/{id}/emotional-state to read the result.
    """
    if not _redis_client or not _redis_client.redis:
        raise HTTPException(status_code=503, detail="Redis unavailable.")

    r = _redis_client.redis
    background_tasks.add_task(_run_analysis, conversation_id, r)

    return {
        "conversation_id": conversation_id,
        "status":          "analysis_started",
        "result_endpoint": f"/conversation/{conversation_id}/emotional-state",
    }


class SessionRequest(BaseModel):
    mode: str = Field("continue", pattern="^(fresh|continue)$")


@router.post("/conversation/{conversation_id}/session",
             dependencies=[Depends(get_current_user)])
async def set_conversation_session(conversation_id: str, req: SessionRequest):
    """
    Control session continuity after an idle period.

    mode=continue (default): preserve the existing emotional state — the bot
      remembers the conversation's emotional arc.

    mode=fresh: reset emotional state to neutral — the bot treats the next
      message as the start of a new emotional context, even within the same
      conversation thread.
    """
    if not _redis_client or not _redis_client.redis:
        raise HTTPException(status_code=503, detail="Redis unavailable.")

    r = _redis_client.redis

    if req.mode == "fresh":
        pipe = r.pipeline()
        pipe.delete(f"conversation:{conversation_id}:emotional_state")
        pipe.delete(f"conv:{conversation_id}:mood_arc")
        pipe.delete(f"conv:{conversation_id}:valence_hist")
        await pipe.execute()
        # Delete per-speaker CDM keys (pattern scan — context_engine per-speaker state)
        async for key in r.scan_iter(f"conv:{conversation_id}:spk:*"):
            await r.delete(key)
        logger.info(
            "session_reset",
            extra={"event": "session_reset", "conversation_id": conversation_id},
        )
        return {"conversation_id": conversation_id, "mode": "fresh", "status": "reset"}

    return {"conversation_id": conversation_id, "mode": "continue", "status": "preserved"}


@router.get("/conversation/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, limit: int = 50,
                                     current_user: dict = Depends(get_current_user)):
    """Get messages with their latest emotion analysis for a conversation."""
    user_id = current_user["sub"]
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            is_member = await conn.fetchval(
                "SELECT 1 FROM conversation_participants WHERE conversation_id = $1 AND user_id = $2",
                conversation_id, user_id
            )
            if not is_member:
                raise HTTPException(status_code=403, detail="Access denied.")

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return []
