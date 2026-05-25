from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import time
import uuid

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.models import MessageEvent
from shared.utils.logger import get_logger
from shared.utils.auth import validate_api_key, RateLimiter

logger = get_logger("ingestion_service")

app = FastAPI(title="Ingestion Service", version="1.0.0")

_allowed_origins = [
    o.strip() for o in
    os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

redis_client = RedisClient()
rate_limiter = RateLimiter(redis_client)

class MessageInput(BaseModel):
    conversation_id: str
    user_id: str
    text: str
    metadata: dict = {}

@app.on_event("startup")
async def startup_event():
    logger.info("Ingestion Service starting...")
    await redis_client.connect()
    logger.info("Ingestion Service ready.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Ingestion Service shutting down...")
    await redis_client.close()

@app.post("/messages")
async def ingest_message(msg: MessageInput, api_key: str = Depends(validate_api_key)):
    # rate limit by api_key instead of spoofable user_id
    if not await rate_limiter.is_allowed(api_key):
        logger.warning(f"Rate limit exceeded for api_key: {api_key[:8]}..., conv={msg.conversation_id}")
        raise HTTPException(status_code=429, detail="Too Many Requests: Rate limit exceeded")
    
    logger.debug(f"Received ingestion request: conv={msg.conversation_id}, user={msg.user_id}, len={len(msg.text)}")
    
    event = MessageEvent(
        conversation_id=msg.conversation_id,
        user_id=msg.user_id,
        text=msg.text,
        timestamp=time.time(),
        message_id=str(uuid.uuid4()),
        metadata=msg.metadata
    )
    
    try:
        # write to stream
        await redis_client.publish_event("message_stream", event.dict())
        logger.info(f"Successfully ingested message {event.message_id} [conv={event.conversation_id}]")
        return {"status": "accepted", "message_id": event.message_id}
    except Exception as e:
        logger.log_exception("CRITICAL INGESTION FAILURE", e)
        raise HTTPException(status_code=500, detail="Service Unavailable: Failed to persist event stream")

@app.get("/health")
async def health():
    return {"status": "ok"}
