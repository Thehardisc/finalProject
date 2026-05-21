"""
persistence_service/db/setup.py — Database initialisation and self-healing schema migrations.
"""
import traceback

from sqlalchemy import text
from shared.utils.logger import get_logger

logger = get_logger("persistence_service")


def init_db(engine):
    """
    Create all tables and apply safe ALTER TABLE migrations.
    Idempotent — safe to call on every startup.
    """
    print("[persistence_service] init_db() called", flush=True)
    from persistence_service.db_models import Base
    try:
        Base.metadata.create_all(bind=engine)
        print("[persistence_service] Tables created via Base.metadata.create_all()", flush=True)
        logger.info("Database tables ensured (SQLAlchemy create_all).")

        with engine.begin() as conn:
            # Emotion analysis columns
            conn.execute(text("ALTER TABLE emotion_analysis ADD COLUMN IF NOT EXISTS "
                              "ground_truth_emotion VARCHAR(50);"))
            conn.execute(text("ALTER TABLE emotion_analysis ADD COLUMN IF NOT EXISTS "
                              "is_verified BOOLEAN DEFAULT FALSE;"))
            conn.execute(text("ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS "
                              "escalation_score FLOAT DEFAULT 0.0;"))

            # Auth columns
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR;"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                              "role VARCHAR(20) DEFAULT 'user';"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                              "is_active BOOLEAN DEFAULT TRUE;"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login FLOAT;"))

            # Profile columns
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR UNIQUE;"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name VARCHAR;"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name VARCHAR;"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR;"))

            # Legacy
            conn.execute(text("ALTER TABLE users ALTER COLUMN username DROP NOT NULL;"))

            # Group chat support
            conn.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS name VARCHAR;"))
            conn.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS creator_user_id VARCHAR;"))

            logger.info("Self-healing schema check complete (including auth columns).")

        print("[persistence_service] init_db() SUCCESS", flush=True)

    except Exception as e:
        print(f"[persistence_service] CRITICAL DB INIT FAILURE: {e}", flush=True)
        traceback.print_exc()
        logger.log_exception("CRITICAL DATABASE INITIALIZATION FAILURE", e)
        logger.warning("Attempting to continue despite migration error.")
