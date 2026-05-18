"""
api_service/routes/users.py — User listing endpoint.
"""
from fastapi import APIRouter, HTTPException, Query

from shared.utils.logger import get_logger
from api_service.db.pool import get_pool

logger = get_logger("api_service")
router = APIRouter()


@router.get("/users")
async def get_users(current_user_id: str = Query(None)):
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not ready")
    async with pool.acquire() as conn:
        if current_user_id:
            rows = await conn.fetch(
                "SELECT user_id, display_name FROM users "
                "WHERE user_id != $1 AND is_active = TRUE",
                current_user_id
            )
        else:
            rows = await conn.fetch(
                "SELECT user_id, display_name FROM users WHERE is_active = TRUE"
            )
    return [dict(r) for r in rows]
