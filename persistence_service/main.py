import asyncio
import sys
import os
import json
import time
import traceback
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from shared.utils.logger import get_logger
from persistence_service.db_models import Base, Message, EmotionAnalysis, ConversationState

print("[persistence_service] Module imports OK", flush=True)
logger = get_logger("persistence_service")

# Database setup
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "emotion_db")
DB_HOST = os.getenv("DB_HOST", "db")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Ensure tables exist
# In production, use Alembic for migrations
# We wait for DB to be ready in docker-compose, but retries here are good.
def init_db():
    print("[persistence_service] init_db() called", flush=True)
    try:
        # Create tables
        # Use Base metadata directly instead of Alembic to guarantee creation
        Base.metadata.create_all(bind=engine)
        print("[persistence_service] Tables created via Base.metadata.create_all()", flush=True)
        logger.info("Database tables ensured (SQLAlchemy create_all).")

        # Add missing columns if they don't exist yet
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE emotion_analysis ADD COLUMN IF NOT EXISTS ground_truth_emotion VARCHAR(50);"))
            conn.execute(text("ALTER TABLE emotion_analysis ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT FALSE;"))
            conn.execute(text("ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS escalation_score FLOAT DEFAULT 0.0;"))
            logger.info("Self-healing schema check complete.")

        print("[persistence_service] init_db() SUCCESS", flush=True)
    except Exception as e:
        print(f"[persistence_service] CRITICAL DB INIT FAILURE: {e}", flush=True)
        traceback.print_exc()
        logger.log_exception("CRITICAL DATABASE INITIALIZATION FAILURE", e)
        logger.warning("Attempting to continue despite migration error.")

redis_client = RedisClient()

# We need to listen to multiple streams
STREAMS = {
    "message_stream": "persistence_group",
    "emotion_stream": "persistence_group", 
    "conversation_update_stream": "persistence_group",
    "feedback_stream": "persistence_group"
}
CONSUMER_NAME = "worker_1"

async def process_message_event(session, data):
    msg = Message(
        message_id=data.get("message_id"),
        conversation_id=data.get("conversation_id"),
        user_id=data.get("user_id"),
        text=data.get("text"),
        timestamp=float(data.get("timestamp", 0))
    )
    session.merge(msg) # use merge to avoid conflicts

async def process_emotion_event(session, data):
    # Store emotion analysis result
    analysis = EmotionAnalysis(
        message_id=data.get("message_id"),
        emotions_json=data.get("emotions", "{}"),
        reasoning_json=data.get("reasoning"),
        pipeline_log_json=data.get("pipeline_log")
    )
    session.add(analysis)

async def process_state_event(session, data):
    # Update conversation state
    conversation_id = data.get("conversation_id")
    state_json = data.get("conversation_state", "{}")
    state_obj = json.loads(state_json)
    
    # Get escalation score
    escalation_score = calculate_sentiment_velocity(session, conversation_id)
    
    state_record = session.query(ConversationState).filter(ConversationState.conversation_id == conversation_id).first()
    if not state_record:
        state_record = ConversationState(conversation_id=conversation_id)
        session.add(state_record)
        
    state_record.state_json = state_json
    state_record.escalation_score = float(escalation_score)
    state_record.last_updated = float(state_obj.get("last_updated", 0))
    
def calculate_sentiment_velocity(session, conversation_id: str, limit: int = 5) -> float:
    """
    Get the escalation score by calculating sentiment trend.
    """
    try:
        # get analysis history
        results = session.query(EmotionAnalysis.emotions_json)\
            .join(Message, Message.message_id == EmotionAnalysis.message_id)\
            .filter(Message.conversation_id == conversation_id)\
            .order_by(Message.timestamp.desc())\
            .limit(limit).all()
        
        if len(results) < 2:
            return 0.0
        
        # Tension indices (Anger, Annoyance, Sadness, Grief, Disgust)
        TENSION_LABELS = ['anger', 'annoyance', 'sadness', 'grief', 'disgust', 'fear']
        
        scores = []
        for (json_str,) in reversed(results):
            data = json.loads(json_str)
            # Calculate total tension
            tension = sum(data.get(label, 0.0) for label in TENSION_LABELS)
            scores.append(tension)
            
        # Get slope
        diffs = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        avg_velocity = sum(diffs) / len(diffs) if diffs else 0.0
        
        # Combine final values
        final_score = scores[-1] + (avg_velocity * 0.5)
        return max(0.0, min(1.0, final_score))
    except Exception as e:
        logger.warning(f"Velocity calc failure: {e}")
        return 0.0
 
async def process_feedback_event(session, data):
    msg_id = data.get("message_id")
    label = data.get("ground_truth_emotion")
    
    if msg_id and label:
        # Find the existing analysis record
        analysis = session.query(EmotionAnalysis).filter(EmotionAnalysis.message_id == msg_id).first()
        if analysis:
            analysis.ground_truth_emotion = label
            analysis.is_verified = True
            logger.info(f"Feedback applied to {msg_id}: Label={label}")
        else:
            # create new record if missing
            new_analysis = EmotionAnalysis(
                message_id=msg_id,
                ground_truth_emotion=label,
                is_verified=True,
                emotions_json="{}"
            )
            session.add(new_analysis)
            logger.info(f"New verified analysis created for {msg_id}: Label={label}")

async def main():
    print("[persistence_service] main() loop starting", flush=True)
    # Retry DB connection
    for _ in range(10):
        try:
            init_db()
            break
        except Exception:
            logger.info("Waiting for Database to be ready...")
            time.sleep(2)
            
    await redis_client.connect()
    r = redis_client.redis
    
    # Create groups
    for stream, group in STREAMS.items():
        try:
            await r.xgroup_create(stream, group, mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Error creating group for {stream}: {e}")

    logger.info("Persistence worker started.")
    
    while True:
        try:
            streams_dict = {stream: ">" for stream in STREAMS}
            messages = await r.xreadgroup("persistence_group", CONSUMER_NAME, streams_dict, count=10, block=2000)

            if messages:
                start_time = time.time()  # ← was missing, caused NameError on every batch
                session = SessionLocal()
                # track ack list
                to_ack: list[tuple] = []
                try:
                    for stream, msgs in messages:
                        for message_id, data in msgs:
                            logger.debug(f"Persisting data from stream: {stream}")

                            if stream == "message_stream":
                                await process_message_event(session, data)
                            elif stream == "emotion_stream":
                                await process_emotion_event(session, data)
                            elif stream == "conversation_update_stream":
                                await process_state_event(session, data)
                            elif stream == "feedback_stream":
                                await process_feedback_event(session, data)

                            to_ack.append((stream, message_id))

                    session.commit()

                    # ack after commit
                    for ack_stream, ack_id in to_ack:
                        await r.xack(ack_stream, "persistence_group", ack_id)

                    elapsed = (time.time() - start_time) * 1000
                    logger.info(f"Successfully persisted batch of {len(to_ack)} events in {elapsed:.2f}ms")

                except Exception as e:
                    session.rollback()
                    logger.log_exception("SQL TRANSACTION ROLLBACK — batch NOT ACKed, will retry", e)
                finally:
                    session.close()

        except Exception as e:
            logger.log_exception("PERSISTENCE WORKER FATAL ERROR", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        print("[persistence_service] Starting main()...", flush=True)
        asyncio.run(main())
    except BaseException as e:
        print(f"[persistence_service] FATAL CRASH (BaseException): {type(e).__name__} - {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

