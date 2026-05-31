"""
api_service/routes/admin.py — Admin-only user management endpoints.
"""
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
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
    logger.info(f"Admin {admin['display_name']} deleted user {user_id}")


# ── Pipeline Inspector ────────────────────────────────────────────────────────

@router.get("/recent-analyses")
async def admin_recent_analyses(
    limit: int = Query(default=50, ge=1, le=200),
    conversation_id: Optional[str] = Query(default=None),
    admin: dict = Depends(require_admin),
):
    """List recent messages with emotion analyses. Admin only."""
    pool = get_pool()
    async with pool.acquire() as conn:
        if conversation_id:
            rows = await conn.fetch(
                """
                SELECT m.message_id, m.text, m.timestamp, m.user_id,
                       m.conversation_id, u.display_name,
                       a.emotions_json, a.pipeline_log_json
                FROM messages m
                LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
                LEFT JOIN users u ON m.user_id = u.user_id
                WHERE m.conversation_id = $1
                ORDER BY m.timestamp DESC LIMIT $2
                """,
                conversation_id, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT m.message_id, m.text, m.timestamp, m.user_id,
                       m.conversation_id, u.display_name,
                       a.emotions_json, a.pipeline_log_json
                FROM messages m
                LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
                LEFT JOIN users u ON m.user_id = u.user_id
                ORDER BY m.timestamp DESC LIMIT $1
                """,
                limit,
            )

    results = []
    for row in rows:
        d = dict(row)
        dominant, confidence = "—", None
        if d.get("pipeline_log_json"):
            try:
                pl = json.loads(d["pipeline_log_json"])
                dominant   = pl.get("dominant_selected", "—")
                confidence = pl.get("meta_confidence")
            except Exception:
                pass
        results.append({
            "message_id":      d["message_id"],
            "text":            (d["text"] or "")[:120],
            "timestamp":       d["timestamp"],
            "user_id":         d["user_id"],
            "display_name":    d.get("display_name") or d["user_id"][:8],
            "conversation_id": d["conversation_id"],
            "dominant":        dominant,
            "confidence":      round(confidence * 100, 1) if confidence is not None else None,
            "has_pipeline":    d.get("pipeline_log_json") is not None,
        })
    return results


@router.get("/pipeline/{message_id}")
async def admin_pipeline_detail(
    message_id: str,
    admin: dict = Depends(require_admin),
):
    """Full step-by-step pipeline breakdown for a message. Admin only."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT m.message_id, m.text, m.timestamp, m.user_id,
                   m.conversation_id, u.display_name,
                   a.emotions_json, a.pipeline_log_json, a.reasoning_json,
                   a.ground_truth_emotion, a.is_verified
            FROM messages m
            LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
            LEFT JOIN users u ON m.user_id = u.user_id
            WHERE m.message_id = $1
            """,
            message_id,
        )

        # Fetch 10 previous messages in the conversation for mood trajectory
        prev_rows = await conn.fetch(
            """
            SELECT m.message_id, m.text, m.timestamp, u.display_name,
                   a.pipeline_log_json
            FROM messages m
            LEFT JOIN emotion_analysis a ON m.message_id = a.message_id
            LEFT JOIN users u ON m.user_id = u.user_id
            WHERE m.conversation_id = $1 AND m.timestamp < $2
            ORDER BY m.timestamp DESC LIMIT 10
            """,
            row["conversation_id"] if row else "",
            row["timestamp"] if row else 0,
        ) if row else []

    if not row:
        raise HTTPException(status_code=404, detail="Message not found.")

    d = dict(row)

    def safe_json(s):
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}

    emotions     = safe_json(d.get("emotions_json"))
    pipeline_log = safe_json(d.get("pipeline_log_json"))
    reasoning    = safe_json(d.get("reasoning_json"))
    models       = pipeline_log.get("models", {})

    trajectory = []
    for r in reversed(prev_rows):
        pl = safe_json(r["pipeline_log_json"])
        trajectory.append({
            "message_id":   r["message_id"],
            "text":         (r["text"] or "")[:80],
            "timestamp":    r["timestamp"],
            "display_name": r["display_name"],
            "dominant":     pl.get("dominant_selected", "neutral"),
            "confidence":   pl.get("meta_confidence"),
        })

    return {
        "message": {
            "id":              d["message_id"],
            "text":            d["text"],
            "timestamp":       d["timestamp"],
            "user_id":         d["user_id"],
            "display_name":    d.get("display_name") or d["user_id"][:8],
            "conversation_id": d["conversation_id"],
        },
        "stages": {
            "vader":      models.get("vader", {}),
            "bert":       models.get("basic_bert", {}),
            "goemotions": models.get("go_emotions", {}),
        },
        "decision": {
            "aggregated":    pipeline_log.get("aggregated", {}),
            "dominant":      pipeline_log.get("dominant_selected"),
            "confidence":    pipeline_log.get("meta_confidence"),
            "decision_mode": pipeline_log.get("decision_mode", "rule-based"),
            "logic_map":     pipeline_log.get("logic_map", {}),
            "sarcasm_score": pipeline_log.get("sarcasm_score", 0),
            "conflict":      pipeline_log.get("conflict"),
        },
        "context": {
            "reasoning":    reasoning,
            "raw_emotions": emotions,
        },
        "ground_truth": d.get("ground_truth_emotion"),
        "is_verified":  d.get("is_verified", False),
        "trajectory":   trajectory,
    }
