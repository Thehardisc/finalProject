from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, status
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
from shared.utils.logger import get_logger
from shared.utils.auth import validate_api_key, RateLimiter, VALID_API_KEYS
from shared.constants import EMOTION_LABELS
from shared.utils.auth import validate_api_key, RateLimiter, VALID_API_KEYS

logger = get_logger("api_service")

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
        logger.info(f"Client connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Active: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        # Broadcast to all connected clients
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")

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
        "logic_map": pipeline_log.get("logic_map", {})
    }
    
    payload["vibe"] = {
        "valence": conv_state.get("average_valence", 0),
        "top_emotions": [conv_state.get("dominant_emotion", "Neutral")]
    }
    
    await manager.broadcast(payload)

async def _handle_reasoning_update(message_id, data):
    payload = {
        "type": "reasoning",
        "message_id": data.get("message_id"),
        "ai_insight": data.get("ai_insight"),
        "timestamp": float(data.get("timestamp", 0))
    }
    await manager.broadcast(payload)

# --- Background Task for Redis Listener ---
async def redis_listener():
    """Listens to conversation_update_stream and broadcasts updates to WebSocket clients."""
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

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, api_key: str = Query(None)):
    # WebSocket Auth Check (Query Parameter)
    if not api_key or api_key not in VALID_API_KEYS:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logger.warning(f"WebSocket auth failed for {client_id}")
        return

    await manager.connect(websocket)
    rate_limiter = RateLimiter(redis_client)
    
    try:
        while True:
            # Keep alive and listen for client messages (e.g. chat)
            data = await websocket.receive_text()
            
            # Rate Limit Check for WebSocket messages
            if not await rate_limiter.is_allowed(api_key):
                # We don't close the connection, just drop the message and send a warning
                await websocket.send_json({"type": "error", "message": "Rate limit exceeded"})
                continue

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
                logger.log_exception("FAILED TO PROCESS CLIENT WEBSOCKET MESSAGE", e)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected normally.")
    except Exception as e:
        logger.log_exception(f"UNEXPECTED WEBSOCKET ERROR FOR {client_id}", e)
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

@app.get("/conversation/{conversation_id}/state", dependencies=[Depends(validate_api_key)])
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

@app.get("/conversation/{conversation_id}/messages", dependencies=[Depends(validate_api_key)])
async def get_conversation_messages(conversation_id: str, limit: int = 50):
    """Fetch recent messages for a conversation directly from PostgreSQL (CQRS read path)."""
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
            messages.append(msg)
            
        return messages
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return []

@app.post("/message/{message_id}/feedback", dependencies=[Depends(validate_api_key)])
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

@app.get("/analytics/calibration", dependencies=[Depends(validate_api_key)])
async def get_calibration_analytics():
    """Calculate model performance metrics based on verified human feedback."""
    import os
    import asyncpg
    from collections import Counter
    
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'db')}:5432/{os.getenv('POSTGRES_DB', 'emotion_db')}"
    
    try:
        conn = await asyncpg.connect(db_url)
        # Fetch verified records
        query = """
            SELECT ground_truth_emotion, emotions_json
            FROM emotion_analysis
            WHERE is_verified = TRUE
        """
        rows = await conn.fetch(query)
        await conn.close()
        
        if not rows:
            return {"status": "no_data", "message": "Provide more feedback to see calibration stats."}
            
        total_verified = len(rows)
        correct_count = 0
        
        # Per-emotion stats
        tp = Counter() # True Positives
        fp = Counter() # False Positives
        fn = Counter() # False Negatives
        
        # Confusion matrix data: [Actual][Predicted] = Count
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
                
        # Calculate Precision/Recall/F1
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
