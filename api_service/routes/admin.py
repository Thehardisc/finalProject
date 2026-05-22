"""
api_service/routes/admin.py — Admin-only user management endpoints.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from shared.utils.logger import get_logger
from api_service.db.pool import get_pool
from api_service.auth_utils import require_admin

logger = get_logger("api_service")
router = APIRouter(prefix="/admin")


class UpdateUserRequest(BaseModel):
    role:      Optional[str]  = None   # "user" | "admin"
    is_active: Optional[bool] = None


@router.get("/users")
async def admin_list_users(admin: dict = Depends(require_admin)):
    """List all users with stats. Admin only."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id, email, display_name, role, is_active, "
            "created_at, last_login FROM users ORDER BY created_at DESC"
        )
    return [dict(r) for r in rows]


@router.patch("/users/{user_id}")
async def admin_update_user(user_id: str, req: UpdateUserRequest,
                             admin: dict = Depends(require_admin)):
    """Update a user's role or active status. Admin only."""
    if user_id == admin["sub"]:
        raise HTTPException(
            status_code=400,
            detail="Admins cannot modify their own account via this endpoint."
        )

    updates, values = [], []
    if req.role is not None:
        if req.role not in ("user", "admin"):
            raise HTTPException(status_code=400,
                                detail="role must be 'user' or 'admin'.")
        updates.append(f"role = ${len(values)+1}")
        values.append(req.role)
    if req.is_active is not None:
        updates.append(f"is_active = ${len(values)+1}")
        values.append(req.is_active)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")

    values.append(user_id)
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE user_id = ${len(values)}",
            *values
        )
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="User not found.")
    logger.info(f"Admin {admin['username']} updated user {user_id}: "
                f"{req.dict(exclude_none=True)}")
    return {"status": "updated", "user_id": user_id}


@router.delete("/users/{user_id}", status_code=204)
async def admin_delete_user(user_id: str, admin: dict = Depends(require_admin)):
    """Permanently delete a user. Admin only."""
    if user_id == admin["sub"]:
        raise HTTPException(status_code=400,
                            detail="Admins cannot delete their own account.")
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM users WHERE user_id = $1", user_id
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="User not found.")
    logger.info(f"Admin {admin['username']} deleted user {user_id}")
