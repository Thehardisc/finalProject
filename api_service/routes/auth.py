"""
api_service/routes/auth.py — Registration, login, logout and session endpoints.
"""
import os
import re
import time
import uuid

from fastapi import APIRouter, HTTPException, Depends, Response
from pydantic import BaseModel, Field

from shared.utils.logger import get_logger, sanitize_email
from api_service.db.pool import get_pool
from api_service.auth_utils import (
    hash_password, verify_password, create_jwt, get_current_user, JWT_EXPIRY_HOURS
)

logger = get_logger("api_service")
router = APIRouter(prefix="/auth")

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")


class RegisterRequest(BaseModel):
    email:      str = Field(..., min_length=3,  max_length=255)
    first_name: str = Field(..., min_length=1,  max_length=50)
    last_name:  str = Field(..., min_length=1,  max_length=50)
    password:   str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    email:    str
    password: str


@router.post("/register", status_code=201)
async def register(req: RegisterRequest, response: Response):
    """Register a new user account with bcrypt-hashed password."""
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', req.email):
        raise HTTPException(status_code=400, detail="Invalid email format.")

    pool = get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT user_id FROM users WHERE email = $1", req.email
        )
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered.")

        user_id      = str(uuid.uuid4())
        pw_hash      = hash_password(req.password)
        role         = "admin" if req.email == ADMIN_USERNAME else "user"
        display_name = f"{req.first_name.strip()} {req.last_name.strip()}"

        await conn.execute(
            "INSERT INTO users "
            "(user_id, email, first_name, last_name, display_name, "
            "password_hash, role, is_active, created_at) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,TRUE,$8)",
            user_id, req.email, req.first_name, req.last_name,
            display_name, pw_hash, role, time.time()
        )
        logger.bind(user_id=user_id, email_hash=sanitize_email(req.email), role=role).info(
            "user_registered",
            extra={"event": "user_registered"},
        )
        token = create_jwt(user_id, display_name, role)
        response.set_cookie(
            key="_req_sid", value=token, httponly=True,
            samesite="lax", max_age=JWT_EXPIRY_HOURS * 3600
        )
        return {"user_id": user_id, "display_name": display_name,
                "email": req.email, "role": role}


@router.post("/login")
async def auth_login(req: LoginRequest, response: Response):
    """Authenticate user and return a signed JWT cookie."""
    pool = get_pool()
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT user_id, email, display_name, password_hash, role, is_active "
            "FROM users WHERE email = $1",
            req.email
        )

    audit = logger.bind(email_hash=sanitize_email(req.email))
    if not user:
        audit.warning("login_failed", extra={"event": "login_failed", "reason": "user_not_found"})
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not user["is_active"]:
        audit.bind(user_id=user["user_id"]).warning(
            "login_failed", extra={"event": "login_failed", "reason": "account_inactive"},
        )
        raise HTTPException(status_code=403,
                            detail="Account has been deactivated. Contact an admin.")
    if not user["password_hash"] or not verify_password(req.password, user["password_hash"]):
        audit.bind(user_id=user["user_id"]).warning(
            "login_failed", extra={"event": "login_failed", "reason": "password_mismatch"},
        )
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET last_login = $1 WHERE user_id = $2",
            time.time(), user["user_id"]
        )

    token = create_jwt(user["user_id"], user["display_name"], user["role"])
    audit.bind(user_id=user["user_id"], role=user["role"]).info(
        "user_login", extra={"event": "user_login"},
    )
    response.set_cookie(
        key="_req_sid", value=token, httponly=True,
        samesite="lax", max_age=JWT_EXPIRY_HOURS * 3600
    )
    return {"user_id":      user["user_id"],
            "display_name": user["display_name"],
            "email":        user["email"],
            "role":         user["role"]}


@router.post("/logout")
async def auth_logout(response: Response):
    """Clear the authentication cookie."""
    response.delete_cookie(key="_req_sid", samesite="lax")
    return {"status": "logged_out"}


@router.get("/me")
async def auth_me(current_user: dict = Depends(get_current_user)):
    """Return profile of the currently authenticated user."""
    pool = get_pool()
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT user_id, email, display_name, role, is_active, "
            "created_at, last_login FROM users WHERE user_id = $1",
            current_user["sub"]
        )
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return dict(user)
