from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import sys
import os
import json

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient

app = FastAPI(title="Emotion API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = RedisClient()

@app.on_event("startup")
async def startup_event():
    await redis_client.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()

@app.get("/conversation/{conversation_id}/state")
async def get_conversation_state(conversation_id: str):
    r = redis_client.redis
    state = await r.hgetall(f"conversation:{conversation_id}")
    if not state:
        # Return a default empty state instead of 404
        return {
            "message_count": 0,
            "overall_mood": "Neutral",
            "average_valence": 0.0,
            "conversation_id": conversation_id,
            "status": "New"
        }
    return state

@app.get("/conversation/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, limit: int = 50):
    """
    Fetch recent messages for a conversation from the database.
    Since API Service doesn't connect to DB directly in this microservices pattern 
    (usually Persistence Service handles DB), we have two options:
    1. Connect API to DB directly for reads (CQRS pattern, common for performance).
    2. Request via Redis (too complex for MVP).
    3. We will connect API to Postgres directly for READS.
    """
    import os
    import asyncpg
    
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'db')}:5432/{os.getenv('POSTGRES_DB', 'emotion_db')}"
    
    try:
        conn = await asyncpg.connect(db_url)
        # Fetch messages with their latest analysis
        # We join messages with emotion_analysis
        query = """
            SELECT m.message_id as id, m.text as content, m.timestamp, m.user_id as sender_id, a.emotions_json as emotions
            FROM messages m
            LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
            WHERE m.conversation_id = $1
            ORDER BY m.timestamp DESC
            LIMIT $2
        """
        rows = await conn.fetch(query, conversation_id, limit)
        await conn.close()
        
        messages = []
        for row in rows:
            msg = dict(row)
            # Timestamp is already a float (unix epoch), which the frontend JS can handle (new Date(ts * 1000) or if it's ms)
            # Python time.time() is seconds. JS Date needs ms usually, or just passing seconds to some parser.
            # But wait, looking at ingestion, we store time.time() which is float seconds.
            # Let's just pass it as is. Frontend `new Date(msg.timestamp * 1000)` might be needed if it expects ms.
            messages.append(msg)
            
        return messages
    except Exception as e:
        print(f"DB Error: {e}")
        return []

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("api_service/static/index.html", "r") as f:
        return f.read()

# Mount static files
app.mount("/static", StaticFiles(directory="api_service/static"), name="static")
