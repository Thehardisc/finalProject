"""
auth_utils.py — Core authentication utilities for InnerLink API service.

Provides:
  - bcrypt password hashing and verification
  - JWT creation and decoding (HS256, signed with JWT_SECRET)
  - FastAPI dependencies: get_current_user, require_admin
"""
import os
import time
import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, Header, status
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-jwt-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.environ.get("JWT_EXPIRY_HOURS", "24"))

# ── Password Hashing ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """
    Hash a plain-text password using bcrypt.
    bcrypt automatically generates and embeds a unique salt per hash.
    Returns a UTF-8 string suitable for DB storage.
    """
    salt = bcrypt.gensalt(rounds=12)  # rounds=12 is the production-grade default
    hashed = bcrypt.hashpw(plain.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Safely compare a plain-text password against a stored bcrypt hash.
    Uses a constant-time comparison to prevent timing attacks.
    """
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ── JWT Tokens ────────────────────────────────────────────────────────────────

def create_jwt(user_id: str, username: str, role: str) -> str:
    """
    Create a signed HS256 JWT containing user identity and role.
    Token expires after JWT_EXPIRY_HOURS hours.
    """
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + (JWT_EXPIRY_HOURS * 3600),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> dict:
    """
    Decode and validate a JWT. Raises HTTPException on failure.
    Returns the decoded payload dict.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI Dependencies ───────────────────────────────────────────────────────

async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    FastAPI dependency — extracts and validates the Bearer JWT from the
    Authorization header. Returns the decoded token payload as a dict
    with keys: sub (user_id), username, role.

    Usage:
        @app.get("/me")
        async def me(user = Depends(get_current_user)):
            ...
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1]
    return decode_jwt(token)


async def require_admin(authorization: Optional[str] = Header(None)) -> dict:
    """
    FastAPI dependency — same as get_current_user but additionally enforces
    that the authenticated user has role == 'admin'.
    Returns the decoded token payload.

    Usage:
        @app.get("/admin/users")
        async def admin_users(user = Depends(require_admin)):
            ...
    """
    user = await get_current_user(authorization)
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )
    return user
