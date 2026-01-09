from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="Ingestion Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = RedisClient()

class MessageInput(BaseModel):
    conversation_id: str
    user_id: str
    text: str
    metadata: dict = {}

@app.on_event("startup")
async def startup_event():
    await redis_client.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()

@app.post("/messages")
async def ingest_message(msg: MessageInput):
    event = MessageEvent(
        conversation_id=msg.conversation_id,
        user_id=msg.user_id,
        text=msg.text,
        timestamp=time.time(),
        message_id=str(uuid.uuid4()),
        metadata=msg.metadata
    )
    
    try:
        # Publish to 'message_stream'
        await redis_client.publish_event("message_stream", event.dict())
        return {"status": "accepted", "message_id": event.message_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
