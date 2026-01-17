from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import sys
import os
import json
import asyncio
import time
import uuid
from typing import List

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

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Active: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        # Broadcast to all connected clients
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to client: {e}")

manager = ConnectionManager()

# --- Background Task for Redis Listener ---
async def redis_listener():
    print("Starting Redis Listener for WebSockets...")
    # We need a separate connection for blocking reads usually, 
    # but our redis_client uses a pool so it might be okay.
    # However, xread is blocking. It's safer to use a dedicated client or careful async.
    # We will use the existing client but ensure we don't block other API calls if possible.
    # Actually, asyncio architecture allows this.
    
    # We want to listen to the final output. 
    # The 'aggregation_service' publishes to 'conversation_update_stream'
    STREAM_KEY = "conversation_update_stream" 
    # Also 'emotion_stream' from central responder if we want raw live updates?
    # User UI shows "Live Analysis", so we might want the stream from Central Responder directly?
    # The Aggregation Service output contains the state update.
    # The Central Responder output contains the immediate analysis.
    # Let's listen to 'emotion_stream' (from Central Responder) for live feedback
    # AND 'conversation_update_stream' (from Aggregation) for vibe updates?
    
    # Let's listen to BOTH or merge them?
    # For now, let's listen to 'emotion_stream' as that is the "Live" event.
    # But wait, the frontend needs "Vibe" stats which come from Aggregation.
    
    # We will listen to 'conversation_update_stream'. 
    # NOTE: The Central Responder pushes to 'emotion_stream', Aggregation reads 'emotion_stream' 
    # and pushes to 'conversation_update_stream'. 
    # So 'conversation_update_stream' happens AFTER aggregation.
    # That event contains "conversation_state". Does it contain the raw analysis details too?
    # In 'aggregation_service/main.py', output_event = data.copy() + conversation_state.
    # So yes, it carries everything.
    
    r = redis_client.redis
    last_id = "$"
    
    while True:
        try:
            # We use xread() here to just tail the stream
            # xread returns [[stream_name, [[message_id, data], ...]]]
            response = await r.xread({STREAM_KEY: last_id}, count=1, block=100)
            
            if response:
                for stream, messages in response:
                    for message_id, data in messages:
                        last_id = message_id
                        
                        # Parse data for frontend
                        # data values are strings.
                        payload = {
                            "type": "analysis",
                            "data": {},
                            "vibe": {}
                        }
                        
                        # Extract Analysis
                        # The fields might be: 'original_text', 'emotions', 'pipeline_log', 'dominant_emotion'
                        # plus 'conversation_state'.
                        
                        raw_text = data.get("original_text", "") or data.get("text", "")
                        
                        # Pipeline Log has details
                        pipeline_log = json.loads(data.get("pipeline_log", "{}"))
                        
                        # Dominant
                        dom_emo = data.get("dominant_emotion", "Neutral")
                        
                        # Conversation State (Vibe)
                        conv_state = json.loads(data.get("conversation_state", "{}"))
                        
                        # Construct Payload matching User's Frontend expectation
                        # User JS expects:
                        # payload.data = { id, raw_text, final_dominant_emotion, final_valence, bert_emotions, llm_insights ... }
                        
                        # We map our data to this structure
                        
                        # BERT emotions
                        # We have 'emotions' map in data.
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
                            "llm_insights": "Analysis Complete", # Placeholder
                            "llm_sarcasm_score": 0.0, # Placeholder
                            "hierarchical_scores": [], # Placeholder
                            "emojis_found": [], # Placeholder: Extract from text?
                            "slang_detected": {}
                        }
                        
                        # Vibe
                        payload["vibe"] = {
                            "valence": conv_state.get("average_valence", 0),
                            "sync_score": 0.8, # Placeholder
                            "resonance": 0.7, # Placeholder
                            "volatility": 0.2, # Placeholder
                            "top_emotions": [conv_state.get("dominant_emotion", "Neutral")]
                        }
                        
                        await manager.broadcast(payload)
                        
        except Exception as e:
            # print(f"Redis Listener Error: {e}")
            await asyncio.sleep(1)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            # Keep alive and listen for client messages (e.g. chat)
            data = await websocket.receive_text()
            # If client sends a chat message, we need to inject it into Ingestion Service
            # The user JS sends: { text, recipient_id: 'system' }
            try:
                msg_obj = json.loads(data)
                text = msg_obj.get("text")
                if text:
                    # Prefer explicit sender_id from client, fallback to client_id
                    sender = msg_obj.get("sender_id", client_id)
                    
                    event = {
                        "text": text,
                        "conversation_id": "conv-1", 
                        "user_id": sender,
                        "timestamp": time.time(),
                        "message_id": str(uuid.uuid4()),
                        "metadata": {"source": "websocket"}
                    }
                    await redis_client.publish_event("message_stream", event)
                    
            except Exception as e:
                print(f"Error processing client msg: {e}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    await redis_client.connect()
    # Start background listener
    asyncio.create_task(redis_listener())

@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()

# Existing Endpoints...

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
            SELECT m.message_id as id, m.text as content, m.timestamp, m.user_id as sender_id, 
                   a.emotions_json as emotions, a.reasoning_json as reasoning, a.pipeline_log_json as pipeline_log
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
