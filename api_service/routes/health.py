"""
api_service/routes/health.py — System readiness health check endpoint.
"""
import os
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from shared.utils.logger import get_logger
from api_service.db.pool import get_pool

logger = get_logger("api_service")
router = APIRouter()


@router.get("/health/status")
async def get_system_status(redis_client=None):
    """
    Returns readiness status for all subsystems.
    Used by the frontend loading gate.
    Returns 503 if not ready, 200 if all systems are online.
    """
    ready_marker = "/app/models/.ready"
    meta_ready   = os.path.exists(ready_marker)

    redis_ok = False
    try:
        pool = get_pool()
        if redis_client and redis_client.redis:
            await redis_client.redis.ping()
            redis_ok = True
    except Exception:
        pass

    db_ok = False
    try:
        pool = get_pool()
        if pool:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1 FROM messages LIMIT 0")
                db_ok = True
    except Exception:
        pass

    training_in_progress = False
    try:
        if redis_client and redis_client.redis:
            flag = await redis_client.redis.get("system:training_in_progress")
            training_in_progress = flag == "1"
    except Exception:
        pass

    all_ready = meta_ready and redis_ok and db_ok

    payload = {
        "ready":                all_ready,
        "timestamp":            time.time(),
        "status":               "online" if all_ready else "warming_up",
        "training_in_progress": training_in_progress,
        "components": {
            "database":    db_ok,
            "redis":       redis_ok,
            "meta_learner": meta_ready,
        }
    }

    if not all_ready:
        return JSONResponse(content=payload, status_code=503)
    return payload
