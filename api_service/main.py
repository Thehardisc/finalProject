"""
api_service/main.py — FastAPI application entry point.

Responsibilities:
  - Create the FastAPI app with CORS
  - Register all route blueprints
  - Handle startup / shutdown lifecycle (DB pool, Redis, WebSocket listener)
  - Register the /ws WebSocket endpoint
  - Mount static files and dashboard
"""
import asyncio
import json
import os
import sys
import time
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from shared.utils.auth import RateLimiter

from api_service.auth_utils import decode_jwt, get_current_user
from api_service.db.pool import init_pool, close_pool, get_pool
from api_service.websocket.manager import manager
from api_service.websocket.listener import redis_listener

# Route blueprints
from api_service.routes import health, auth, admin, users, conversations, messages, analytics
import api_service.routes.conversations as conv_routes
import api_service.routes.messages     as msg_routes

logger = get_logger("api_service")

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Emotion API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost",
                   "http://127.0.0.1", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = RedisClient()
rate_limiter  = None

MAX_MESSAGE_LENGTH = int(os.environ.get("MAX_MESSAGE_LENGTH", 2000))

# ── Register routes ────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(users.router)
app.include_router(conversations.router)
app.include_router(messages.router)
app.include_router(analytics.router)

# Health route needs redis_client injected — we wrap it here
@app.get("/health/status")
async def health_status():
    return await health.get_system_status(redis_client=redis_client)


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
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
    try:
        while True:
            data = await websocket.receive_text()
            if rate_limiter and not await rate_limiter.is_allowed(user_id):
                await websocket.send_json({"type": "error", "message": "Rate limit exceeded"})
                continue
            try:
                msg_obj = json.loads(data)
                text    = msg_obj.get("text")
                if text:
                    if len(text) > MAX_MESSAGE_LENGTH:
                        await websocket.send_json({
                            "type":    "error",
                            "message": f"Message too long (max {MAX_MESSAGE_LENGTH} characters)."
                        })
                        continue
                    sender          = msg_obj.get("sender_id", user_id)
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
                        "metadata":        {"source": "websocket"}
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


# ── Lifecycle ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global rate_limiter
    await init_pool()
    await redis_client.connect()
    rate_limiter = RateLimiter(redis_client)

    # Inject redis into route modules that need it
    conv_routes.set_redis(redis_client)
    msg_routes.set_redis(redis_client)

    asyncio.create_task(redis_listener(redis_client))


@app.on_event("shutdown")
async def shutdown_event():
    await close_pool()
    await redis_client.close()


# ── Static / Dashboard ─────────────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("api_service/static/index.html", "r") as f:
        return f.read()

app.mount("/static", StaticFiles(directory="api_service/static"), name="static")
