"""
api_service/websocket/listener.py — Redis → WebSocket broadcast bridge.

Listens to conversation_update_stream and reasoning_update_stream, then
pushes the payloads to the relevant connected WebSocket clients.
"""
import json
import asyncio

from shared.utils.logger import get_logger
from api_service.websocket.manager import manager
from api_service.db.pool import get_pool

logger = get_logger("api_service")


async def _handle_conversation_update(message_id, data):
    raw_text     = data.get("original_text", "") or data.get("text", "")
    pipeline_log = json.loads(data.get("pipeline_log", "{}"))
    dom_emo      = data.get("dominant_emotion", "Neutral")
    conv_state   = json.loads(data.get("conversation_state", "{}"))
    ems          = json.loads(data.get("emotions", "{}"))

    bert_list = [
        {"label": k, "score": float(v)}
        for k, v in ems.items()
        if k not in ['vader_neg', 'vader_neu', 'vader_pos',
                     'vader_compound', 'dominant_emotion']
    ]

    payload = {
        "type": "analysis",
        "data": {
            "id":                    str(message_id),
            "raw_text":              raw_text,
            "final_dominant_emotion": dom_emo,
            "final_valence":         float(ems.get("vader_compound", 0)),
            "bert_emotions":         bert_list,
            "meta_confidence":       float(pipeline_log.get("meta_confidence", 0.0)),
            "context_shift":         json.loads(data.get("context_shift", "null")),
            "logic_map":             pipeline_log.get("logic_map", {}),
            "sender_id":             data.get("user_id")
        },
        "vibe": {
            "valence":     conv_state.get("average_valence", 0),
            "top_emotions": [conv_state.get("dominant_emotion", "Neutral")]
        }
    }

    convo_id = data.get("conversation_id")
    if not convo_id:
        logger.warning("Conversation update with no conversation_id — skipping broadcast.")
        return

    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id FROM conversation_participants WHERE conversation_id = $1",
            convo_id
        )
        for r in rows:
            await manager.broadcast_to_user(r["user_id"], payload)


async def _handle_reasoning_update(message_id, data):
    payload = {
        "type":       "reasoning",
        "message_id": data.get("message_id"),
        "ai_insight": data.get("ai_insight"),
        "timestamp":  float(data.get("timestamp", 0))
    }
    msg_id = data.get("message_id")
    pool   = get_pool()
    if pool and msg_id:
        async with pool.acquire() as conn:
            convo_id = await conn.fetchval(
                "SELECT conversation_id FROM messages WHERE message_id = $1", msg_id
            )
            if convo_id:
                rows = await conn.fetch(
                    "SELECT user_id FROM conversation_participants "
                    "WHERE conversation_id = $1",
                    convo_id
                )
                for r in rows:
                    await manager.broadcast_to_user(r["user_id"], payload)


async def redis_listener(redis_client) -> None:
    """Listen to Redis streams and push updates to connected WebSocket clients."""
    logger.info("Starting Redis Listener for WebSockets...")
    r = redis_client.redis
    STREAM_KEYS = ["conversation_update_stream", "reasoning_update_stream"]
    last_ids = {k: "$" for k in STREAM_KEYS}

    while True:
        try:
            response = await r.xread(last_ids, count=1, block=100)
            if response:
                for stream, messages in response:
                    for message_id, data in messages:
                        stream_name = stream.decode() if isinstance(stream, bytes) else stream
                        last_ids[stream_name] = message_id

                        if stream_name == "conversation_update_stream":
                            await _handle_conversation_update(message_id, data)
                        elif stream_name == "reasoning_update_stream":
                            await _handle_reasoning_update(message_id, data)
        except Exception as e:
            logger.log_exception("WebSocket Redis Listener Error", e)
            await asyncio.sleep(1)
