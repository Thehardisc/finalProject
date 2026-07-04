"""api_service/demo_users.py — Pre-generated demo users and their upsert logic."""
import os
import time
import uuid

from api_service.auth_utils import hash_password

DEMO_USERS = [
    {"user_id": "531c7f56-e5c4-4557-9b2d-e7e8ed7c942f", "email": "alice@demo.innerlink",   "first_name": "Alice",   "last_name": "Chen",  "role": "admin"},
    {"user_id": "90b04411-2879-4e2d-adb9-cf254793d1d2", "email": "bob@demo.innerlink",     "first_name": "Bob",     "last_name": "Kim"},
    {"user_id": "b3c15d22-3f4a-4b8e-a1c9-df365804e3a1", "email": "charlie@demo.innerlink", "first_name": "Charlie", "last_name": "Park"},
    {"user_id": "c4d26e33-4f5b-4c9f-b2da-ef476915f4b2", "email": "diana@demo.innerlink",   "first_name": "Diana",   "last_name": "Lee"},
    {"user_id": "d5e37f44-5f6c-4d0f-c3eb-f0587a26f5c3", "email": "eve@demo.innerlink",     "first_name": "Eve",     "last_name": "Zhao"},
]
_DEMO_PW = os.environ.get("DEMO_PASSWORD", "demo-innerlink-2026")


# Ensure DEMO_USERS[slot] exists in the users table; returns the user dict.
async def upsert_demo_user(conn, slot: int) -> dict:
    demo = DEMO_USERS[slot]
    row = await conn.fetchrow(
        "SELECT user_id, display_name, role FROM users WHERE email = $1", demo["email"]
    )
    desired_role = demo.get("role", "user")
    if row:
        user_id      = row["user_id"]
        display_name = row["display_name"]
        role         = row["role"]
        if role != desired_role:
            await conn.execute(
                "UPDATE users SET role = $1 WHERE user_id = $2", desired_role, user_id
            )
            role = desired_role
    else:
        user_id      = demo.get("user_id") or str(uuid.uuid4())
        display_name = f"{demo['first_name']} {demo['last_name']}"
        pw_hash      = hash_password(_DEMO_PW)
        role         = desired_role
        await conn.execute(
            """INSERT INTO users
               (user_id, email, first_name, last_name, display_name,
                password_hash, role, is_active, created_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,TRUE,$8)""",
            user_id, demo["email"], demo["first_name"], demo["last_name"],
            display_name, pw_hash, role, time.time()
        )
    await conn.execute(
        "UPDATE users SET last_login = $1 WHERE user_id = $2", time.time(), user_id
    )
    return {"user_id": user_id, "display_name": display_name, "email": demo["email"], "role": role}
