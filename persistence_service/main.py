import asyncio
import sys
import os
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils.redis_client import RedisClient
from persistence_service.db_models import Base, Message, EmotionAnalysis, ConversationState

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
    try:
        # Run blind migrations first to ensure schema matches code
        run_migrations()
        Base.metadata.create_all(bind=engine)
        print("Database tables created.")
    except Exception as e:
        print(f"Error creating tables: {e}")

redis_client = RedisClient()

# We need to listen to multiple streams
STREAMS = {
    "message_stream": "persistence_group", # For raw messages
    "emotion_stream": "persistence_group", # For analysis results
    "conversation_update_stream": "persistence_group" # For state updates
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
    session.merge(msg) # usage of merge to avoid duplicates if re-processed

async def process_emotion_event(session, data):
    # This event extends message event but has emotions
    # We store the analysis result
    analysis = EmotionAnalysis(
        message_id=data.get("message_id"),
        emotions_json=data.get("emotions", "{}"),
        reasoning_json=data.get("reasoning"),
        pipeline_log_json=data.get("pipeline_log")
    )
    session.add(analysis)

# "Poor Man's Migration" to ensure columns exist without wiping DB
def run_migrations():
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("ALTER TABLE emotion_analysis ADD COLUMN IF NOT EXISTS reasoning_json TEXT;"))
            conn.execute(text("ALTER TABLE emotion_analysis ADD COLUMN IF NOT EXISTS pipeline_log_json TEXT;"))
            # Depending on sqlalchemy version / driver, we might need to commit
            try:
                conn.commit()
            except:
                pass # Auto-commit in some drivers
            print("Transformations/Migrations applied successfully.")
    except Exception as e:
        print(f"Migration / Column addition warning: {e}")

async def process_state_event(session, data):
    # Update conversation state
    conversation_id = data.get("conversation_id")
    state_json = data.get("conversation_state", "{}")
    state_obj = json.loads(state_json)
    
    state_record = ConversationState(
        conversation_id=conversation_id,
        state_json=state_json,
        last_updated=float(state_obj.get("last_updated", 0))
    )
    session.merge(state_record)

async def main():
    # Retry DB connection
    for _ in range(10):
        try:
            init_db()
            break
        except Exception:
            print("Waiting for DB...")
            time.sleep(2)
            
    await redis_client.connect()
    r = redis_client.redis
    
    # Create groups
    for stream, group in STREAMS.items():
        try:
            await r.xgroup_create(stream, group, mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                print(f"Error creating group for {stream}: {e}")

    print("Persistence worker started...")
    
    while True:
        try:
            # We can read from multiple streams at once
            streams_dict = {stream: ">" for stream in STREAMS}
            # Note: xreadgroup allows reading from multiple streams
            messages = await r.xreadgroup("persistence_group", CONSUMER_NAME, streams_dict, count=10, block=2000)
            
            if messages:
                session = SessionLocal()
                try:
                    for stream, msgs in messages:
                        for message_id, data in msgs:
                            print(f"Persisting data from {stream}")
                            
                            if stream == "message_stream":
                                await process_message_event(session, data)
                            elif stream == "emotion_stream":
                                await process_emotion_event(session, data)
                            elif stream == "conversation_update_stream":
                                await process_state_event(session, data)
                            
                            await r.xack(stream, "persistence_group", message_id)
                    
                    session.commit()
                except Exception as e:
                    session.rollback()
                    print(f"Error persisting batch: {e}")
                finally:
                    session.close()
            
        except Exception as e:
            print(f"Error in processing loop: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
