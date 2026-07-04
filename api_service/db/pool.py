"""api_service/db/pool.py — asyncpg connection pool lifecycle management."""
import os
import asyncio

from shared.utils.logger import get_logger

logger = get_logger("api_service")

_pool = None


def get_pool():
    """Return the shared asyncpg pool. Raises if not yet initialised."""
    if _pool is None:
        raise RuntimeError("DB pool has not been initialised yet.")
    return _pool


async def init_pool() -> None:
    """Create the asyncpg pool with retry logic. Called from startup_event."""
    global _pool
    import asyncpg
    db_url = (
        f"postgresql://{os.getenv('POSTGRES_USER','user')}:"
        f"{os.getenv('POSTGRES_PASSWORD','password')}@"
        f"{os.getenv('DB_HOST','db')}:5432/"
        f"{os.getenv('POSTGRES_DB','emotion_db')}"
    )
    for attempt in range(5):
        try:
            _pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
            logger.info("DB pool created successfully.")
            return
        except Exception as e:
            logger.warning(f"DB pool attempt {attempt+1}/5 failed: {e}. Retrying in 3s...")
            await asyncio.sleep(3)
    logger.error("CRITICAL: Could not create DB pool after 5 attempts.")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
