from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import sys
import os
import json
import asyncio
import time
import uuid
from typing import List, Optional

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from shared.utils.auth import validate_api_key, RateLimiter, VALID_API_KEYS
from shared.constants import EMOTION_LABELS
from api_service.auth_utils import hash_password, verify_password, create_jwt, decode_jwt, get_current_user, require_admin, JWT_EXPIRY_HOURS

logger = get_logger("api_service")

app = FastAPI(title="Emotion API", version="1.0.0")

@app.get("/health/status")
async def get_system_status():
    """
    Returns readiness status for subsystems.
    Used by frontend to know when to stop loading.
    Returns 503 if not ready, 200 if OK.
    """
    ready_marker = "/app/models/.ready"
    meta_ready = os.path.exists(ready_marker)

    # Check Redis
    redis_ok = False
    try:
        if redis_client.redis:
            await redis_client.redis.ping()
            redis_ok = True
    except Exception:
        pass

    # Check DB — reuse the shared pool, don't open a new connection every poll
    db_ok = False
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1 FROM messages LIMIT 0")
                db_ok = True
    except Exception:
        pass

    all_ready = meta_ready and redis_ok and db_ok

    # Check if trainer is still in its first cycle by reading a Redis flag
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost", "http://127.0.0.1", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = RedisClient()
rate_limiter = None

# Websocket Manager
db_pool = None

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
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to {user_id}: {e}")

manager = ConnectionManager()

async def _handle_conversation_update(message_id, data):
    payload = {
        "type": "analysis",
        "data": {},
        "vibe": {}
    }
    
    raw_text = data.get("original_text", "") or data.get("text", "")
    pipeline_log = json.loads(data.get("pipeline_log", "{}"))
    dom_emo = data.get("dominant_emotion", "Neutral")
    conv_state = json.loads(data.get("conversation_state", "{}"))
    ems = json.loads(data.get("emotions", "{}"))
    
    bert_list = []
    for k, v in ems.items():
        if k not in ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'dominant_emotion']:
            bert_list.append({"label": k, "score": float(v)})
    
    payload["data"] = {
        "id": str(message_id),
        "raw_text": raw_text,
        "final_dominant_emotion": dom_emo,
        "final_valence": float(ems.get("vader_compound", 0)),
        "bert_emotions": bert_list,
        "meta_confidence": float(pipeline_log.get("meta_confidence", 0.0)),
        "context_shift": json.loads(data.get("context_shift", "null")),
        "logic_map": pipeline_log.get("logic_map", {}),
        "sender_id": data.get("user_id")
    }
    
    payload["vibe"] = {
        "valence": conv_state.get("average_valence", 0),
        "top_emotions": [conv_state.get("dominant_emotion", "Neutral")]
    }
    
    convo_id = data.get("conversation_id")
    if not convo_id:
        logger.warning("Received conversation update with no conversation_id — skipping broadcast.")
        return
    if db_pool:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT user_id FROM conversation_participants WHERE conversation_id = $1", convo_id)
            for r in rows:
                await manager.broadcast_to_user(r["user_id"], payload)

async def _handle_reasoning_update(message_id, data):
    payload = {
        "type": "reasoning",
        "message_id": data.get("message_id"),
        "ai_insight": data.get("ai_insight"),
        "timestamp": float(data.get("timestamp", 0))
    }
    msg_id = data.get("message_id")
    if db_pool and msg_id:
        async with db_pool.acquire() as conn:
            convo_id = await conn.fetchval("SELECT conversation_id FROM messages WHERE message_id = $1", msg_id)
            if convo_id:
                rows = await conn.fetch("SELECT user_id FROM conversation_participants WHERE conversation_id = $1", convo_id)
                for r in rows:
                    await manager.broadcast_to_user(r["user_id"], payload)

# Listener for Redis
async def redis_listener():
    """Listen to updates and broadcast to clients."""
    logger.info("Starting Redis Listener for WebSockets...")
    
    r = redis_client.redis
    STREAM_KEYS = ["conversation_update_stream", "reasoning_update_stream"]
    last_ids = {k: "$" for k in STREAM_KEYS}
    
    while True:
        try:
            # Multi-stream read
            response = await r.xread(last_ids, count=1, block=100)
            
            if response:
                for stream, messages in response:
                    for message_id, data in messages:
                        last_ids[stream.decode() if isinstance(stream, bytes) else stream] = message_id
                        
                        stream_name = stream.decode() if isinstance(stream, bytes) else stream
                        
                        if stream_name == "conversation_update_stream":
                            await _handle_conversation_update(message_id, data)
                        elif stream_name == "reasoning_update_stream":
                            await _handle_reasoning_update(message_id, data)
                        
        except Exception as e:
            logger.log_exception("WebSocket Redis Listener Error", e)
            await asyncio.sleep(1)

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    # WebSocket Auth Check via Cookie
    token = websocket.cookies.get("_req_sid")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logger.warning(f"WebSocket auth failed: missing cookie for {user_id}")
        return
    try:
        user_data = decode_jwt(token)
        if user_data["sub"] != user_id:
            raise Exception("User ID mismatch")
    except Exception as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logger.warning(f"WebSocket auth failed for {user_id}: {e}")
        return

    await manager.connect(websocket, user_id)
    # Use module-level rate_limiter (not per-connection instantiation)
    
    try:
        while True:
            # Keep alive and listen for client messages (e.g. chat)
            data = await websocket.receive_text()
            
            # Rate Limit Check for WebSocket messages
            # Rate-limit by user_id, not api_key, so users don't share the same bucket.
            if rate_limiter and not await rate_limiter.is_allowed(user_id):
                await websocket.send_json({"type": "error", "message": "Rate limit exceeded"})
                continue

            # handle incoming messages
            # formats: { text, recipient_id: 'system' }
            try:
                msg_obj = json.loads(data)
                text = msg_obj.get("text")
                if text:
                    sender = msg_obj.get("sender_id", user_id)
                    # use existing conversation id or create new one
                    conversation_id = msg_obj.get("conversation_id") or user_id

                    event = {
                        "text": text,
                        "conversation_id": conversation_id,
                        "user_id": sender,
                        "timestamp": time.time(),
                        "message_id": str(uuid.uuid4()),
                        "metadata": {"source": "websocket"}
                    }
                    await redis_client.publish_event("message_stream", event)
            except Exception as e:
                logger.log_exception("FAILED TO PROCESS CLIENT WEBSOCKET MESSAGE", e)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"Client {user_id} disconnected normally.")
    except Exception as e:
        logger.log_exception(f"UNEXPECTED WEBSOCKET ERROR FOR {user_id}", e)
        manager.disconnect(websocket, user_id)

@app.on_event("startup")
async def startup_event():
    global db_pool, rate_limiter
    import asyncpg
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'db')}:5432/{os.getenv('POSTGRES_DB', 'emotion_db')}"
    # Retry DB pool creation — DB may not be ready immediately
    for attempt in range(5):
        try:
            db_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
            logger.info("DB pool created successfully.")
            break
        except Exception as e:
            logger.warning(f"DB pool attempt {attempt+1}/5 failed: {e}. Retrying in 3s...")
            await asyncio.sleep(3)
    else:
        logger.error("CRITICAL: Could not create DB pool after 5 attempts. DB-dependent features will fail.")

    await redis_client.connect()
    rate_limiter = RateLimiter(redis_client)
    # Start background listener
    asyncio.create_task(redis_listener())

@app.on_event("shutdown")
async def shutdown_event():
    global db_pool
    if db_pool:
        await db_pool.close()
    await redis_client.close()

# ── Auth Endpoints ─────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=8)

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/register", status_code=201)
async def register(req: RegisterRequest, response: Response):
    """Register a new user account with bcrypt-hashed password."""
    import re
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', req.email):
        raise HTTPException(status_code=400, detail="Invalid email format.")

    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT user_id FROM users WHERE email = $1", req.email)
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered.")

        user_id = str(uuid.uuid4())
        pw_hash = hash_password(req.password)
        # First registered user OR matches ADMIN_USERNAME gets admin role
        role = "admin" if req.email == ADMIN_USERNAME else "user"
        display_name = f"{req.first_name.strip()} {req.last_name.strip()}"

        await conn.execute(
            """INSERT INTO users (user_id, email, first_name, last_name, display_name, password_hash, role, is_active, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, TRUE, $8)""",
            user_id, req.email, req.first_name, req.last_name, display_name, pw_hash, role, time.time()
        )
        logger.info(f"New user registered: {req.email} (role={role})")
        token = create_jwt(user_id, display_name, role)
        response.set_cookie(
            key="_req_sid",
            value=token,
            httponly=True,
            samesite="lax",
            max_age=JWT_EXPIRY_HOURS * 3600
        )
        return {"user_id": user_id, "display_name": display_name, "email": req.email, "role": role}


@app.post("/auth/login")
async def auth_login(req: LoginRequest, response: Response):
    """Authenticate user with password, return signed JWT."""
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT user_id, email, display_name, password_hash, role, is_active FROM users WHERE email = $1",
            req.email
        )

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account has been deactivated. Contact an admin.")
    if not user["password_hash"] or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Update last_login timestamp
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET last_login = $1 WHERE user_id = $2", time.time(), user["user_id"])

    token = create_jwt(user["user_id"], user["display_name"], user["role"])
    logger.info(f"User logged in: {user['email']}")
    response.set_cookie(
        key="_req_sid",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=JWT_EXPIRY_HOURS * 3600
    )
    return {"user_id": user["user_id"], "display_name": user["display_name"], "email": user["email"], "role": user["role"]}

@app.post("/auth/logout")
async def auth_logout(response: Response):
    """Clear the authentication cookie."""
    response.delete_cookie(key="_req_sid", samesite="lax")
    return {"status": "logged_out"}


@app.get("/auth/me")
async def auth_me(current_user: dict = Depends(get_current_user)):
    """Return profile of the currently authenticated user."""
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT user_id, email, display_name, role, is_active, created_at, last_login FROM users WHERE user_id = $1",
            current_user["sub"]
        )
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return dict(user)


# ── Admin Endpoints ────────────────────────────────────────────────────────────

class UpdateUserRequest(BaseModel):
    role: Optional[str] = None       # "user" | "admin"
    is_active: Optional[bool] = None


@app.get("/admin/users")
async def admin_list_users(admin: dict = Depends(require_admin)):
    """List all users with stats. Admin only."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id, email, display_name, role, is_active, created_at, last_login FROM users ORDER BY created_at DESC"
        )
    return [dict(r) for r in rows]


@app.patch("/admin/users/{user_id}")
async def admin_update_user(user_id: str, req: UpdateUserRequest, admin: dict = Depends(require_admin)):
    """Update a user's role or active status. Admin only."""
    if user_id == admin["sub"]:
        raise HTTPException(status_code=400, detail="Admins cannot modify their own account via this endpoint.")

    updates = []
    values = []
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
            f"UPDATE users SET {', '.join(updates)} WHERE user_id = ${len(values)}",
            *values
        )
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="User not found.")
    logger.info(f"Admin {admin['username']} updated user {user_id}: {req.dict(exclude_none=True)}")
    return {"status": "updated", "user_id": user_id}


@app.delete("/admin/users/{user_id}", status_code=204)
async def admin_delete_user(user_id: str, admin: dict = Depends(require_admin)):
    """Permanently delete a user. Admin only."""
    if user_id == admin["sub"]:
        raise HTTPException(status_code=400, detail="Admins cannot delete their own account.")
    async with db_pool.acquire() as conn:
        result = await conn.execute("DELETE FROM users WHERE user_id = $1", user_id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="User not found.")
    logger.info(f"Admin {admin['username']} deleted user {user_id}")
    return

@app.get("/users/online")
async def get_online_users():
    """Return user IDs that currently have an active WebSocket connection."""
    return {"online_user_ids": list(manager.active_connections.keys())}


DEMO_USERS = [
    {"email": "alice@demo.innerlink",   "first_name": "Alice",   "last_name": "Chen"},
    {"email": "bob@demo.innerlink",     "first_name": "Bob",     "last_name": "Kim"},
    {"email": "charlie@demo.innerlink", "first_name": "Charlie", "last_name": "Park"},
    {"email": "diana@demo.innerlink",   "first_name": "Diana",   "last_name": "Lee"},
    {"email": "eve@demo.innerlink",     "first_name": "Eve",     "last_name": "Zhao"},
]
_DEMO_PW = "demo-innerlink-2026"


@app.post("/auth/demo-login/{slot}")
async def demo_login(slot: int, response: Response):
    """One-click login as a preset demo user (slot 0-4). Creates the account on first use."""
    if slot < 0 or slot >= len(DEMO_USERS):
        raise HTTPException(status_code=400, detail="Invalid demo slot.")
    demo = DEMO_USERS[slot]
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id, display_name, role FROM users WHERE email = $1", demo["email"]
        )
        if row:
            user_id      = row["user_id"]
            display_name = row["display_name"]
            role         = row["role"]
        else:
            user_id      = str(uuid.uuid4())
            display_name = f"{demo['first_name']} {demo['last_name']}"
            pw_hash      = hash_password(_DEMO_PW)
            role         = "user"
            await conn.execute(
                """INSERT INTO users
                   (user_id, email, first_name, last_name, display_name,
                    password_hash, role, is_active, created_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,TRUE,$8)""",
                user_id, demo["email"], demo["first_name"], demo["last_name"],
                display_name, pw_hash, role, time.time()
            )
        await conn.execute(
            "UPDATE users SET last_login = $1 WHERE user_id = $2", time.time(), user_id
        )

    token = create_jwt(user_id, display_name, role)
    response.set_cookie(
        key="_req_sid", value=token,
        httponly=True, samesite="lax",
        max_age=JWT_EXPIRY_HOURS * 3600,
    )
    return {"user_id": user_id, "display_name": display_name,
            "email": demo["email"], "role": role}


@app.get("/users")
async def get_users(current_user_id: str = Query(None)):
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not ready")
    async with db_pool.acquire() as conn:
        if current_user_id:
            rows = await conn.fetch("SELECT user_id, display_name FROM users WHERE user_id != $1 AND is_active = TRUE", current_user_id)
        else:
            rows = await conn.fetch("SELECT user_id, display_name FROM users WHERE is_active = TRUE")
    return [dict(r) for r in rows]

class CreateConversationRequest(BaseModel):
    user_id: str
    target_user_id: str

@app.post("/conversations")
async def create_conversation(req: CreateConversationRequest):
    import asyncpg
    import time
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'db')}:5432/{os.getenv('POSTGRES_DB', 'emotion_db')}"
    conn = await asyncpg.connect(db_url)
    try:
        query = """
            SELECT c.conversation_id 
            FROM conversations c
            JOIN conversation_participants p1 ON c.conversation_id = p1.conversation_id
            JOIN conversation_participants p2 ON c.conversation_id = p2.conversation_id
            WHERE c.type = 'direct' 
              AND p1.user_id = $1 
              AND p2.user_id = $2
        """
        existing = await conn.fetchval(query, req.user_id, req.target_user_id)
        if existing:
            return {"conversation_id": existing}
            
        conv_id = f"conv-{str(uuid.uuid4())[:8]}"
        async with conn.transaction():
            await conn.execute("INSERT INTO conversations (conversation_id, type, created_at) VALUES ($1, 'direct', $2)", conv_id, time.time())
            await conn.execute("INSERT INTO conversation_participants (conversation_id, user_id, joined_at) VALUES ($1, $2, $3)", conv_id, req.user_id, time.time())
            await conn.execute("INSERT INTO conversation_participants (conversation_id, user_id, joined_at) VALUES ($1, $2, $3)", conv_id, req.target_user_id, time.time())
            
        return {"conversation_id": conv_id}
    finally:
        await conn.close()

@app.get("/conversations/{user_id}")
async def my_conversations(user_id: str):
    import asyncpg
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'db')}:5432/{os.getenv('POSTGRES_DB', 'emotion_db')}"
    conn = await asyncpg.connect(db_url)
    try:
        query = """
            SELECT c.conversation_id, c.type, c.created_at, u.display_name as other_display_name, u.user_id as other_user_id
            FROM conversations c
            JOIN conversation_participants my_p ON my_p.conversation_id = c.conversation_id
            JOIN conversation_participants other_p ON other_p.conversation_id = c.conversation_id AND other_p.user_id != $1
            JOIN users u ON other_p.user_id = u.user_id
            WHERE my_p.user_id = $1
            ORDER BY c.created_at DESC
        """
        rows = await conn.fetch(query, user_id)
        
        # enrich with state
        r = redis_client.redis
        result = []
        for row in rows:
            d = dict(row)
            state = await r.hgetall(f"conversation:{d['conversation_id']}")
            if state:
                d['average_valence'] = float(state.get("average_valence", 0.0))
                d['dominant_emotion'] = state.get("dominant_emotion", "Neutral")
            else:
                d['average_valence'] = 0.0
                d['dominant_emotion'] = "Neutral"
            result.append(d)
        return result
    finally:
        await conn.close()

@app.get("/conversation/{conversation_id}/state", dependencies=[Depends(get_current_user)])
async def get_conversation_state(conversation_id: str):
    r = redis_client.redis
    state = await r.hgetall(f"conversation:{conversation_id}")
    if not state:
        # return default state
        return {
            "message_count": 0,
            "overall_mood": "Neutral",
            "average_valence": 0.0,
            "conversation_id": conversation_id,
            "status": "New"
        }
    return state

@app.get("/conversation/{conversation_id}/messages", dependencies=[Depends(get_current_user)])
async def get_conversation_messages(conversation_id: str, limit: int = 50):
    """Get messages for a conversation."""
    import os
    import asyncpg
    
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'db')}:5432/{os.getenv('POSTGRES_DB', 'emotion_db')}"
    
    try:
        conn = await asyncpg.connect(db_url)
        try:
            # get messages and latest analysis
            query = """
                SELECT m.message_id as id, m.text as content, m.timestamp, m.user_id as sender_id, 
                       a.emotions_json as emotions, a.reasoning_json as reasoning, a.pipeline_log_json as pipeline_log
                FROM messages m
                LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
                WHERE m.conversation_id = $1
                ORDER BY m.timestamp DESC
                LIMIT $2
            """
            rows = await conn.fetch(query, conversation_id, limit)
        finally:
            await conn.close()
        
        messages = []
        for row in rows:
            msg = dict(row)
            messages.append(msg)
            
        return messages
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return []

@app.post("/message/{message_id}/feedback", dependencies=[Depends(get_current_user)])
async def post_message_feedback(message_id: str, payload: dict):
    label = payload.get("label")
    if not label or label not in EMOTION_LABELS:
        raise HTTPException(status_code=400, detail=f"Invalid emotion label. Must be one of: {EMOTION_LABELS}")
    
    event = {
        "message_id": message_id,
        "ground_truth_emotion": label,
        "timestamp": time.time()
    }
    
    try:
        await redis_client.publish_event("feedback_stream", event)
        logger.info(f"Feedback received for {message_id}: {label}")
        return {"status": "accepted", "message_id": message_id}
    except Exception as e:
        logger.error(f"Failed to publish feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/analytics/calibration", dependencies=[Depends(get_current_user)])
async def get_calibration_analytics():
    """Get model performance metrics."""
    import os
    import asyncpg
    from collections import Counter
    
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'db')}:5432/{os.getenv('POSTGRES_DB', 'emotion_db')}"
    
    try:
        conn = await asyncpg.connect(db_url)
        try:
            # get verified records
            query = """
                SELECT ground_truth_emotion, emotions_json
                FROM emotion_analysis
                WHERE is_verified = TRUE
            """
            rows = await conn.fetch(query)
        finally:
            await conn.close()
        
        if not rows:
            return {"status": "no_data", "message": "Provide more feedback to see calibration stats."}
            
        total_verified = len(rows)
        correct_count = 0
        
        # Per-emotion stats
        tp = Counter() # True Positives
        fp = Counter() # False Positives
        fn = Counter() # False Negatives
        
        # confusion matrix: [Actual][Predicted] = count
        confusion = {} 
        
        for row in rows:
            actual = row['ground_truth_emotion']
            ems = json.loads(row['emotions_json'])
            predicted = ems.get("dominant_emotion", "Neutral")
            
            # Init confusion matrix rows
            if actual not in confusion: confusion[actual] = Counter()
            confusion[actual][predicted] += 1
            
            if actual == predicted:
                correct_count += 1
                tp[actual] += 1
            else:
                fp[predicted] += 1
                fn[actual] += 1
                
        # calc stats
        emotion_stats = {}
        for emo in EMOTION_LABELS:
            # We only show stats for emotions that appear in the verified set
            if actual_count := sum(1 for r in rows if r['ground_truth_emotion'] == emo):
                precision = tp[emo] / (tp[emo] + fp[emo]) if (tp[emo] + fp[emo]) > 0 else 0
                recall = tp[emo] / (tp[emo] + fn[emo]) if (tp[emo] + fn[emo]) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                emotion_stats[emo] = {
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "samples": actual_count
                }
        
        logger.log_stats("Model Calibration Report", {
            "Total Samples": total_verified,
            "Overall Accuracy": f"{correct_count / total_verified:.2%}",
            "TP Total": sum(tp.values()),
            "FP Total": sum(fp.values()),
            "FN Total": sum(fn.values())
        })

        return {
            "overall_accuracy": round(correct_count / total_verified, 4),
            "total_verified_samples": total_verified,
            "emotion_breakdown": emotion_stats,
            "confusion_matrix": confusion,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Analytics DB Error: {e}")
        raise HTTPException(status_code=500, detail="Could not calculate analytics.")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("api_service/static/index.html", "r") as f:
        return f.read()

# Mount static files
app.mount("/static", StaticFiles(directory="api_service/static"), name="static")
