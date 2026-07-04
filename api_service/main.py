from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, status, Response
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import HTMLResponse
import sys
import os
import json
import asyncio
import time
import uuid
from collections import OrderedDict
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger, sanitize_email
from shared.utils.auth import RateLimiter
from shared.constants import EMOTION_LABELS

_NON_EMOTION = frozenset({
    'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'dominant_emotion',
})
_EMOTION_SET = frozenset(EMOTION_LABELS)
from api_service.auth_utils import hash_password, verify_password, create_jwt, decode_jwt, get_current_user, require_admin, JWT_EXPIRY_HOURS
from api_service.db.pool import init_pool as _init_pool, close_pool as _close_pool, get_pool
from api_service.routes.conversations import router as conv_router, set_redis as _conv_set_redis, set_cache_invalidator as _conv_set_cache_invalidator
from api_service.routes.messages import router as msg_router, set_redis as _msg_set_redis
from api_service.routes.ai_demo import router as ai_demo_router, set_redis as _ai_demo_set_redis
from api_service.demo_users import DEMO_USERS, upsert_demo_user

logger = get_logger("api_service")

app = FastAPI(title="Emotion API", version="1.0.0")

app.include_router(conv_router)
app.include_router(msg_router)
app.include_router(ai_demo_router)

_DEFAULT_ORIGINS = "http://localhost:5173,http://localhost,http://127.0.0.1,http://127.0.0.1:5173"
_allowed_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = RedisClient()
rate_limiter = None
db_pool      = None


@app.get("/health")
@app.get("/health/status")
async def get_system_status():
    """Returns readiness status for subsystems. Returns 503 if not ready."""
    ready_marker = "/app/models/.ready"
    meta_ready = os.path.exists(ready_marker)

    redis_ok = False
    try:
        if redis_client.redis:
            await redis_client.redis.ping()
            redis_ok = True
    except Exception:
        pass

    db_ok = False
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                await asyncio.wait_for(
                    conn.fetchval("SELECT 1 FROM messages LIMIT 0"), timeout=5.0
                )
                db_ok = True
    except Exception:
        pass

    all_ready = meta_ready and redis_ok and db_ok

    training_in_progress = False
    try:
        if redis_client.redis:
            flag = await redis_client.redis.get("system:training_in_progress")
            training_in_progress = flag == "1"
    except Exception:
        pass

    payload = {
        "ready": all_ready,
        "timestamp": time.time(),
        "status": "online" if all_ready else "warming_up",
        "training_in_progress": training_in_progress,
        "components": {
            "database": db_ok,
            "redis": redis_ok,
            "meta_learner": meta_ready,
        }
    }

    if not all_ready:
        from fastapi.responses import JSONResponse
        return JSONResponse(content=payload, status_code=503)

    return payload


@app.get("/health/model")
async def get_model_health():
    """Returns meta-learner stats: gate weights, accuracy, calibration temperature."""
    try:
        stats = await redis_client.redis.hgetall("model:stats") if redis_client.redis else {}
    except Exception:
        stats = {}

    if not stats:
        return {"status": "training", "message": "Model not yet deployed."}

    return {
        "status":                    stats.get("status", "unknown"),
        "model_version":             stats.get("model_version"),
        "feature_dim":               int(stats.get("feature_dim", 0)),
        "test_accuracy":             float(stats.get("test_accuracy", 0)),
        "test_f1_macro":             float(stats.get("test_f1_macro", 0)),
        "calibration_temperature":   float(stats.get("calibration_temperature", 1.0)),
        "gate_alpha": {
            "vader":   float(stats.get("vader_gate", 0)),
            "bert":    float(stats.get("bert_gate", 0)),
            "goe":     float(stats.get("goe_gate", 0)),
            "context": float(stats.get("ctx_gate", 0)),
        },
        "training_samples":          int(stats.get("training_samples", 0)),
        "last_trained_utc":          stats.get("last_trained_utc"),
    }


MONITORED_STREAMS = [
    "message_stream",
    "preprocessed_stream",
    "partial_analysis_stream",
    "emotion_stream",
    "conversation_update_stream",
]

@app.get("/metrics/streams")
async def get_stream_depths():
    """Returns current XLEN for each Redis stream. Use as an early warning for consumer lag."""
    if not redis_client.redis:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "redis unavailable"}, status_code=503)
    try:
        depths = {}
        for stream in MONITORED_STREAMS:
            try:
                depths[stream] = await redis_client.redis.xlen(stream)
            except Exception:
                depths[stream] = None
        return {"timestamp": time.time(), "streams": depths, "maxlen": int(os.getenv("REDIS_STREAM_MAXLEN", 10_000))}
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": str(e)}, status_code=500)


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        logger.info(f"Client {user_id} connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            try:
                self.active_connections[user_id].remove(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            except ValueError:
                pass
        logger.info(f"Client {user_id} disconnected. Active: {len(self.active_connections)}")

    async def broadcast_to_user(self, user_id: str, message: dict):
        if user_id not in self.active_connections:
            return
        dead = []
        for connection in list(self.active_connections[user_id]):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to {user_id}: {e}")
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn, user_id)
        if dead:
            logger.info(f"Removed {len(dead)} stale connection(s) for user {user_id}.")


manager = ConnectionManager()

_participant_cache: OrderedDict = OrderedDict()
_PARTICIPANT_CACHE_MAX = 1000


async def _get_participants(pool, conv_id: str) -> list:
    if conv_id in _participant_cache:
        _participant_cache.move_to_end(conv_id)
        return _participant_cache[conv_id]
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id FROM conversation_participants WHERE conversation_id = $1", conv_id
        )
    user_ids = [r["user_id"] for r in rows]
    _participant_cache[conv_id] = user_ids
    if len(_participant_cache) > _PARTICIPANT_CACHE_MAX:
        _participant_cache.popitem(last=False)
    return user_ids


async def _handle_conversation_update(message_id, data):
    raw_text     = data.get("original_text", "") or data.get("text", "")
    pipeline_log = json.loads(data.get("pipeline_log", "{}"))
    dom_emo      = data.get("dominant_emotion", "Neutral")
    conv_state   = json.loads(data.get("conversation_state", "{}"))
    ems          = json.loads(data.get("emotions", "{}"))
    vader_data   = json.loads(data.get("vader", "{}"))
    convo_id     = data.get("conversation_id")

    mlog = logger.bind(
        message_id=data.get("message_id"),
        conversation_id=convo_id,
        user_id=data.get("user_id"),
    )

    if not convo_id:
        mlog.warning(
            "broadcast_skipped_no_convo_id",
            extra={"event": "broadcast_skipped", "reason": "no_conversation_id"},
        )
        return

    emotion_list = [
        {"label": k, "score": float(v)}
        for k, v in ems.items()
        if k not in _NON_EMOTION and isinstance(v, (int, float))
    ]
    emotion_list.sort(key=lambda x: x["score"], reverse=True)

    payload = {
        "type": "analysis",
        "data": {
            "id":                     data.get("message_id") or str(message_id),
            "conversation_id":        convo_id,
            "raw_text":               raw_text,
            "final_dominant_emotion": dom_emo,
            "final_valence":          float(vader_data.get("vader_compound", 0.0)),
            "bert_emotions":          emotion_list,
            "ekman_group":            pipeline_log.get("ekman_group"),
            "meta_confidence":        float(pipeline_log.get("meta_confidence", 0.0)),
            "context_shift":          json.loads(data.get("context_shift", "null")),
            "logic_map":              pipeline_log.get("logic_map", {}),
            "gate_weights_alpha":     pipeline_log.get("gate_weights_alpha"),
            "sender_id":              data.get("user_id"),
            "context_snapshot":       pipeline_log.get("context_snapshot"),
            "lstm_trajectory":        pipeline_log.get("trajectory"),
            "sarcasm_score":          float(pipeline_log.get("sarcasm_score", 0)),
            "inversion_applied":      bool(pipeline_log.get("inversion_applied", False)),
            "vad":                    pipeline_log.get("vad", {}),
            "dynamics":               pipeline_log.get("dynamics", {}),
            "appraisal":              pipeline_log.get("appraisal") or {},
        },
        "vibe": {
            "valence":     conv_state.get("average_valence", 0),
            "top_emotions": [conv_state.get("dominant_emotion", "Neutral")],
        },
    }

    user_ids = await _get_participants(get_pool(), convo_id)
    for uid in user_ids:
        await manager.broadcast_to_user(uid, payload)
    mlog.debug(
        "ws_broadcast_done",
        extra={"event": "ws_broadcast", "recipients": len(user_ids), "dominant": dom_emo},
    )


async def _handle_reasoning_update(message_id, data):
    payload = {
        "type":       "reasoning",
        "message_id": data.get("message_id"),
        "ai_insight": data.get("ai_insight"),
        "timestamp":  float(data.get("timestamp", 0)),
    }
    msg_id = data.get("message_id")
    if msg_id:
        pool = get_pool()
        async with pool.acquire() as conn:
            convo_id = await conn.fetchval(
                "SELECT conversation_id FROM messages WHERE message_id = $1", msg_id
            )
        if convo_id:
            user_ids = await _get_participants(pool, convo_id)
            for uid in user_ids:
                await manager.broadcast_to_user(uid, payload)


async def _handle_model_ready(message_id, data):
    payload = {
        "type":       "model_ready",
        "message_id": data.get("message_id"),
        "model":      data.get("model"),
    }
    conv_id = data.get("conversation_id")
    if conv_id:
        user_ids = await _get_participants(get_pool(), conv_id)
        for uid in user_ids:
            await manager.broadcast_to_user(uid, payload)


async def _handle_conversation_idle(data: dict) -> None:
    from api_service.routes.conversations import _run_analysis
    cid = data.get("conversation_id", "")
    if not cid:
        return
    try:
        await _run_analysis(cid, redis_client.redis)
        logger.info(
            "idle_analysis_done",
            extra={"event": "idle_analysis_done", "conversation_id": cid},
        )
    except Exception as e:
        logger.warning(f"Idle analysis failed for {cid}: {e}")


async def redis_listener():
    logger.info("Starting Redis Listener for WebSockets...")
    r = redis_client.redis
    STREAM_KEYS = [
        "conversation_update_stream",
        "reasoning_update_stream",
        "partial_result_stream",
        "conversation_idle_stream",
    ]
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
                        elif stream_name == "partial_result_stream":
                            await _handle_model_ready(message_id, data)
                        elif stream_name == "conversation_idle_stream":
                            asyncio.create_task(
                                _handle_conversation_idle(data)
                            )
        except Exception as e:
            logger.log_exception("WebSocket Redis Listener Error", e)
            await asyncio.sleep(1)


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    audit = logger.bind(user_id=user_id)
    token = websocket.cookies.get("_req_sid")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        audit.warning(
            "ws_auth_failed",
            extra={"event": "ws_auth_failed", "reason": "missing_cookie"},
        )
        return
    try:
        user_data = decode_jwt(token)
        if user_data["sub"] != user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            audit.warning(
                "ws_auth_failed",
                extra={"event": "ws_auth_failed", "reason": "user_mismatch",
                       "token_sub": user_data.get("sub")},
            )
            return
    except Exception as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        audit.warning(
            "ws_auth_failed",
            extra={
                "event":       "ws_auth_failed",
                "reason":      "decode_error",
                "error_class": type(e).__name__,
            },
        )
        return

    await manager.connect(websocket, user_id)

    try:
        while True:
            data = await websocket.receive_text()

            if rate_limiter and not await rate_limiter.is_allowed(user_id):
                await websocket.send_json({"type": "error", "message": "Rate limit exceeded"})
                logger.bind(user_id=user_id).warning(
                    "ws_rate_limited",
                    extra={"event": "ws_rate_limited"},
                )
                continue

            try:
                msg_obj = json.loads(data)
                text = msg_obj.get("text")
                if text:
                    sender = (
                        msg_obj.get("sender_id")
                        if user_data.get("role") == "admin" and msg_obj.get("sender_id")
                        else user_id
                    )
                    conversation_id = msg_obj.get("conversation_id")
                    if not conversation_id:
                        await websocket.send_json({"type": "error", "message": "conversation_id is required."})
                        continue

                    event = {
                        "text":            text,
                        "conversation_id": conversation_id,
                        "user_id":         sender,
                        "timestamp":       time.time(),
                        "message_id":      str(uuid.uuid4()),
                        "metadata":        {"source": "websocket"},
                    }
                    await redis_client.publish_event("message_stream", event)
                    logger.bind(
                        message_id=event["message_id"],
                        conversation_id=conversation_id,
                        user_id=sender,
                    ).info(
                        "ws_message_received",
                        extra={"event": "ws_message_received", "text_len": len(text)},
                    )
            except Exception as e:
                logger.bind(user_id=user_id).log_exception("ws_message_processing_failed", e)

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"Client {user_id} disconnected normally.")
    except Exception as e:
        logger.log_exception(f"UNEXPECTED WEBSOCKET ERROR FOR {user_id}", e)
        manager.disconnect(websocket, user_id)


@app.on_event("startup")
async def startup_event():
    global db_pool, rate_limiter
    await _init_pool()
    db_pool = get_pool()

    await redis_client.connect()
    rate_limiter = RateLimiter(redis_client)
    _conv_set_redis(redis_client)
    _msg_set_redis(redis_client)
    _ai_demo_set_redis(redis_client)
    _conv_set_cache_invalidator(lambda conv_id: _participant_cache.pop(conv_id, None))

    asyncio.create_task(redis_listener())


@app.on_event("shutdown")
async def shutdown_event():
    global db_pool
    await _close_pool()
    db_pool = None
    await redis_client.close()


from pydantic import BaseModel, Field

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")


class RegisterRequest(BaseModel):
    email:      str = Field(..., min_length=3, max_length=255)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name:  str = Field(..., min_length=1, max_length=50)
    password:   str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    email:    str
    password: str


@app.post("/auth/register", status_code=201)
async def register(req: RegisterRequest, response: Response):
    import re
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', req.email):
        raise HTTPException(status_code=400, detail="Invalid email format.")

    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT user_id FROM users WHERE email = $1", req.email)
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered.")

        user_id = str(uuid.uuid4())
        pw_hash = hash_password(req.password)
        role = "admin" if req.email == ADMIN_USERNAME else "user"
        display_name = f"{req.first_name.strip()} {req.last_name.strip()}"

        await conn.execute(
            """INSERT INTO users (user_id, email, first_name, last_name, display_name, password_hash, role, is_active, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, TRUE, $8)""",
            user_id, req.email, req.first_name, req.last_name, display_name, pw_hash, role, time.time()
        )
        logger.bind(user_id=user_id, email_hash=sanitize_email(req.email), role=role).info(
            "user_registered",
            extra={"event": "user_registered"},
        )
        token = create_jwt(user_id, display_name, role)
        response.set_cookie(
            key="_req_sid", value=token,
            httponly=True, samesite="lax",
            max_age=JWT_EXPIRY_HOURS * 3600,
        )
        return {"user_id": user_id, "display_name": display_name, "email": req.email, "role": role}


@app.post("/auth/login")
async def auth_login(req: LoginRequest, response: Response):
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT user_id, email, display_name, password_hash, role, is_active FROM users WHERE email = $1",
            req.email
        )

    audit = logger.bind(email_hash=sanitize_email(req.email))
    if not user:
        audit.warning("login_failed", extra={"event": "login_failed", "reason": "user_not_found"})
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not user["is_active"]:
        audit.bind(user_id=user["user_id"]).warning(
            "login_failed", extra={"event": "login_failed", "reason": "account_inactive"},
        )
        raise HTTPException(status_code=403, detail="Account has been deactivated. Contact an admin.")
    if not user["password_hash"] or not verify_password(req.password, user["password_hash"]):
        audit.bind(user_id=user["user_id"]).warning(
            "login_failed", extra={"event": "login_failed", "reason": "password_mismatch"},
        )
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET last_login = $1 WHERE user_id = $2", time.time(), user["user_id"])

    token = create_jwt(user["user_id"], user["display_name"], user["role"])
    audit.bind(user_id=user["user_id"], role=user["role"]).info(
        "user_login", extra={"event": "user_login"},
    )
    response.set_cookie(
        key="_req_sid", value=token,
        httponly=True, samesite="lax",
        max_age=JWT_EXPIRY_HOURS * 3600,
    )
    return {"user_id": user["user_id"], "display_name": user["display_name"], "email": user["email"], "role": user["role"]}


@app.post("/auth/logout")
async def auth_logout(response: Response):
    response.delete_cookie(key="_req_sid", samesite="lax")
    return {"status": "logged_out"}


@app.get("/auth/me")
async def auth_me(current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT user_id, email, display_name, role, is_active, created_at, last_login FROM users WHERE user_id = $1",
            current_user["sub"]
        )
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return dict(user)


class UpdateUserRequest(BaseModel):
    role:      Optional[str]  = None
    is_active: Optional[bool] = None


@app.get("/admin/users")
async def admin_list_users(admin: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id, email, display_name, role, is_active, created_at, last_login FROM users ORDER BY created_at DESC"
        )
    return [dict(r) for r in rows]


@app.patch("/admin/users/{user_id}")
async def admin_update_user(user_id: str, req: UpdateUserRequest, admin: dict = Depends(require_admin)):
    if user_id == admin["sub"]:
        raise HTTPException(status_code=400, detail="Admins cannot modify their own account via this endpoint.")

    updates, values = [], []
    if req.role is not None:
        if req.role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="role must be 'user' or 'admin'.")
        updates.append(f"role = ${len(values)+1}")
        values.append(req.role)
    if req.is_active is not None:
        updates.append(f"is_active = ${len(values)+1}")
        values.append(req.is_active)

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")

    values.append(user_id)
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE user_id = ${len(values)}", *values
        )
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="User not found.")
    logger.bind(
        actor=admin["sub"], target=user_id, action="update_user",
    ).info(
        "admin_action",
        extra={"event": "admin_action", "changes": req.dict(exclude_none=True)},
    )
    return {"status": "updated", "user_id": user_id}


@app.delete("/admin/users/{user_id}", status_code=204)
async def admin_delete_user(user_id: str, admin: dict = Depends(require_admin)):
    if user_id == admin["sub"]:
        raise HTTPException(status_code=400, detail="Admins cannot delete their own account.")
    async with db_pool.acquire() as conn:
        result = await conn.execute("DELETE FROM users WHERE user_id = $1", user_id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="User not found.")
    logger.bind(
        actor=admin["sub"], target=user_id, action="delete_user",
    ).info(
        "admin_action",
        extra={"event": "admin_action"},
    )
    return


@app.get("/admin/recent-analyses")
async def admin_recent_analyses(
    limit: int = 50,
    conversation_id: Optional[str] = None,
    admin: dict = Depends(require_admin),
):
    async with db_pool.acquire() as conn:
        if conversation_id:
            rows = await conn.fetch(
                """
                SELECT m.message_id, m.text, m.timestamp, m.user_id,
                       m.conversation_id, u.display_name,
                       a.pipeline_log_json
                FROM messages m
                LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
                LEFT JOIN users u ON m.user_id = u.user_id
                WHERE m.conversation_id = $1
                ORDER BY m.timestamp DESC LIMIT $2
                """,
                conversation_id, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT m.message_id, m.text, m.timestamp, m.user_id,
                       m.conversation_id, u.display_name,
                       a.pipeline_log_json
                FROM messages m
                LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
                LEFT JOIN users u ON m.user_id = u.user_id
                ORDER BY m.timestamp DESC LIMIT $1
                """,
                limit,
            )

    results = []
    for row in rows:
        d = dict(row)
        dominant, confidence = "—", None
        if d.get("pipeline_log_json"):
            try:
                pl = json.loads(d["pipeline_log_json"])
                dominant   = pl.get("dominant_selected", "—")
                confidence = pl.get("meta_confidence")
            except Exception:
                pass
        results.append({
            "message_id":      d["message_id"],
            "text":            (d["text"] or "")[:120],
            "timestamp":       d["timestamp"],
            "user_id":         d["user_id"],
            "display_name":    d.get("display_name") or d["user_id"][:8],
            "conversation_id": d["conversation_id"],
            "dominant":        dominant,
            "confidence":      round(confidence * 100, 1) if confidence is not None else None,
            "has_pipeline":    d.get("pipeline_log_json") is not None,
        })
    return results


@app.get("/admin/pipeline/{message_id}")
async def admin_pipeline_detail(
    message_id: str,
    admin: dict = Depends(require_admin),
):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT m.message_id, m.text, m.timestamp, m.user_id,
                   m.conversation_id, u.display_name,
                   a.emotions_json, a.pipeline_log_json, a.reasoning_json,
                   a.ground_truth_emotion, a.is_verified
            FROM messages m
            LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
            LEFT JOIN users u ON m.user_id = u.user_id
            WHERE m.message_id = $1
            """,
            message_id,
        )
        prev_rows = await conn.fetch(
            """
            SELECT m.message_id, m.text, m.timestamp, u.display_name,
                   a.pipeline_log_json
            FROM messages m
            LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
            LEFT JOIN users u ON m.user_id = u.user_id
            WHERE m.conversation_id = $1 AND m.timestamp < $2
            ORDER BY m.timestamp DESC LIMIT 10
            """,
            row["conversation_id"] if row else "",
            row["timestamp"] if row else 0,
        ) if row else []

    if not row:
        raise HTTPException(status_code=404, detail="Message not found.")

    d = dict(row)

    def _json(s):
        try:
            return json.loads(s) if s else {}
        except Exception:
            return {}

    emotions     = _json(d.get("emotions_json"))
    pipeline_log = _json(d.get("pipeline_log_json"))
    reasoning    = _json(d.get("reasoning_json"))
    models       = pipeline_log.get("models", {})

    trajectory = []
    for r in reversed(prev_rows):
        pl = _json(r["pipeline_log_json"])
        trajectory.append({
            "message_id":   r["message_id"],
            "text":         (r["text"] or "")[:80],
            "timestamp":    r["timestamp"],
            "display_name": r.get("display_name"),
            "dominant":     pl.get("dominant_selected", "neutral"),
            "confidence":   pl.get("meta_confidence"),
        })

    return {
        "message": {
            "id":              d["message_id"],
            "text":            d["text"],
            "timestamp":       d["timestamp"],
            "user_id":         d["user_id"],
            "display_name":    d.get("display_name") or d["user_id"][:8],
            "conversation_id": d["conversation_id"],
        },
        "stages": {
            "vader":      models.get("vader", {}),
            "bert":       models.get("basic_bert", {}),
            # go_emotions carries side-channel keys (vad_*, goe_confidence) — show only real labels
            "goemotions": {k: v for k, v in models.get("go_emotions", {}).items()
                           if k in _EMOTION_SET},
        },
        "decision": {
            "aggregated":            pipeline_log.get("aggregated", {}),
            "decision_trace":        pipeline_log.get("decision_trace", []),
            "dominant":              pipeline_log.get("dominant_selected"),
            "confidence":            pipeline_log.get("meta_confidence"),
            "decision_mode":         pipeline_log.get("decision_mode", "rule-based"),
            "logic_map":             pipeline_log.get("logic_map", {}),
            "ctx_correction_weight": pipeline_log.get("ctx_correction_weight", 0.0),
            "sarcasm_score":         pipeline_log.get("sarcasm_score", 0),
            "conflict":              pipeline_log.get("conflict"),
            "gate_weights_alpha":    pipeline_log.get("gate_weights_alpha"),
            "ekman_group":           pipeline_log.get("ekman_group"),
        },
        "context": {
            "reasoning":        reasoning,
            "raw_emotions":     emotions,
            "context_snapshot": pipeline_log.get("context_snapshot"),
            "lstm_trajectory":  pipeline_log.get("trajectory"),
            "vad":              pipeline_log.get("vad", {}),
            "dynamics":         pipeline_log.get("dynamics", {}),
            "appraisal":        pipeline_log.get("appraisal", {}),
        },
        "ground_truth": d.get("ground_truth_emotion"),
        "is_verified":  d.get("is_verified", False),
        "trajectory":   trajectory,
    }


@app.get("/users/online", dependencies=[Depends(get_current_user)])
async def get_online_users():
    return {"online_user_ids": list(manager.active_connections.keys())}


@app.post("/auth/demo-login/{slot}")
async def demo_login(slot: int, response: Response):
    if slot < 0 or slot >= len(DEMO_USERS):
        raise HTTPException(status_code=400, detail="Invalid demo slot.")
    async with db_pool.acquire() as conn:
        user = await upsert_demo_user(conn, slot)

    token = create_jwt(user["user_id"], user["display_name"], user["role"])
    response.set_cookie(
        key="_req_sid", value=token,
        httponly=True, samesite="lax",
        max_age=JWT_EXPIRY_HOURS * 3600,
    )
    return user


@app.get("/users", dependencies=[Depends(get_current_user)])
async def get_users(current_user_id: str = Query(None)):
    async with db_pool.acquire() as conn:
        if current_user_id:
            rows = await conn.fetch(
                "SELECT user_id, display_name FROM users WHERE user_id != $1 AND is_active = TRUE "
                "AND email NOT LIKE '%@ai-demo.innerlink'",
                current_user_id
            )
        else:
            rows = await conn.fetch(
                "SELECT user_id, display_name FROM users WHERE is_active = TRUE "
                "AND email NOT LIKE '%@ai-demo.innerlink'"
            )
    return [dict(r) for r in rows]


@app.get("/analytics/calibration", dependencies=[Depends(get_current_user)])
async def get_calibration_analytics():
    from collections import Counter

    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT ground_truth_emotion, emotions_json FROM emotion_analysis WHERE is_verified = TRUE"
            )
    except Exception as e:
        logger.error(f"Analytics DB Error: {e}")
        raise HTTPException(status_code=500, detail="Could not calculate analytics.")

    if not rows:
        return {"status": "no_data", "message": "Provide more feedback to see calibration stats."}

    total_verified = len(rows)
    correct_count  = 0
    tp = Counter()
    fp = Counter()
    fn = Counter()
    confusion = {}

    for row in rows:
        actual    = row['ground_truth_emotion']
        ems       = json.loads(row['emotions_json'])
        numeric   = {k: float(v) for k, v in ems.items() if isinstance(v, (int, float)) and k in EMOTION_LABELS}
        if not numeric:
            total_verified -= 1
            continue
        predicted = max(numeric, key=numeric.get)

        if actual not in confusion:
            confusion[actual] = Counter()
        confusion[actual][predicted] += 1

        if actual == predicted:
            correct_count += 1
            tp[actual] += 1
        else:
            fp[predicted] += 1
            fn[actual] += 1

    emotion_stats = {}
    for emo in EMOTION_LABELS:
        actual_count = sum(1 for r in rows if r['ground_truth_emotion'] == emo)
        if actual_count:
            precision = tp[emo] / (tp[emo] + fp[emo]) if (tp[emo] + fp[emo]) > 0 else 0
            recall    = tp[emo] / (tp[emo] + fn[emo]) if (tp[emo] + fn[emo]) > 0 else 0
            f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            emotion_stats[emo] = {
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "samples":   actual_count,
            }

    if total_verified == 0:
        return {"status": "no_data", "message": "All verified rows are feedback-only. No model outputs to evaluate."}

    logger.log_stats("Model Calibration Report", {
        "Total Samples":    total_verified,
        "Overall Accuracy": f"{correct_count / total_verified:.2%}",
    })

    return {
        "overall_accuracy":       round(correct_count / total_verified, 4),
        "total_verified_samples": total_verified,
        "emotion_breakdown":      emotion_stats,
        "confusion_matrix":       confusion,
        "timestamp":              time.time(),
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("api_service/static/index.html", "r") as f:
        return f.read()
