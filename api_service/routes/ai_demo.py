"""api_service/routes/ai_demo.py — Admin-only AI demo: two Claude agents chat as demo users."""
import asyncio
import os
import time
import uuid

import anthropic
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import random

from shared.utils.logger import get_logger
from api_service.db.pool import get_pool
from api_service.auth_utils import require_admin, hash_password
from api_service.routes.conversations import create_group_conversation, CreateGroupRequest

logger = get_logger("api_service")
router = APIRouter(prefix="/admin/ai-demo", tags=["ai-demo"])

_redis_client = None

def set_redis(client):
    global _redis_client
    _redis_client = client


_MODEL = os.getenv("AI_DEMO_MODEL", "claude-sonnet-5")
_PACING_SECONDS = 3.0

_PERSONA_TEMPLATES = [
    "Upbeat and enthusiastic, wears their heart on their sleeve; gets genuinely "
    "excited, but also frustrated or hurt quickly when dismissed.",
    "Dry, skeptical, a bit sarcastic; pushes back on hype, can get defensive "
    "when challenged, but warms up when met halfway.",
]

_state = {"status": "idle", "conversation_id": None, "topic": None,
          "sent": 0, "total": 0, "error": None}
_task = None
_lock = asyncio.Lock()


class StartAiDemoRequest(BaseModel):
    topic:        str = Field(..., min_length=2, max_length=80)
    num_messages: int = Field(10, ge=4, le=30)


# Agent users are generated per run: random uuid + name, an unguessable password
# (nobody can log in as them), and an @ai-demo.innerlink email that /users filters out.
AGENT_EMAIL_DOMAIN = "ai-demo.innerlink"

_FIRST_NAMES = ["Maya", "Leo", "Nina", "Omar", "Tara", "Felix", "Iris", "Jonas",
                "Lena", "Marco", "Priya", "Sam", "Noa", "Ethan", "Zoe", "Ravi"]
_LAST_NAMES  = ["Rivers", "Stone", "Sato", "Alvarez", "Novak", "Meyer", "Okafor",
                "Lindgren", "Costa", "Haddad", "Kovacs", "Brennan", "Ito", "Moreau"]


async def _create_agent_user(conn, taken_names):
    while True:
        display_name = f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"
        if display_name not in taken_names:
            taken_names.add(display_name)
            break
    first, last = display_name.split(" ", 1)
    user_id = str(uuid.uuid4())
    email   = f"{first.lower()}.{last.lower()}.{user_id[:6]}@{AGENT_EMAIL_DOMAIN}"
    await conn.execute(
        """INSERT INTO users
           (user_id, email, first_name, last_name, display_name,
            password_hash, role, is_active, created_at)
           VALUES ($1,$2,$3,$4,$5,$6,'user',TRUE,$7)""",
        user_id, email, first, last, display_name,
        hash_password(str(uuid.uuid4())), time.time()
    )
    return {"user_id": user_id, "display_name": display_name, "email": email}


def _system_prompt(name, other, persona, topic):
    return (
        f"You are {name}, a real person texting your friend {other} in a chat app. "
        f"You two are talking about: {topic}. Personality: {persona} "
        "Write exactly ONE short chat message (1-2 sentences, under 30 words). "
        "Casual texting tone. Let real emotions show and evolve across the conversation - "
        "excitement, disagreement, annoyance, warmth, reconciliation. "
        "Never mention being an AI. No narration or stage directions. "
        "Output only the message text."
    )


# Kickoff user turn guarantees the messages list never starts with an assistant turn.
def _build_messages(transcript, me, topic):
    messages = [{"role": "user",
                 "content": f"(Conversation start. Topic: {topic}. Send your first message.)"}]
    for speaker, text in transcript:
        messages.append({"role": "assistant" if speaker == me else "user", "content": text})
    return messages


async def _run_ai_demo(conv_id, topic, num_messages, user_a, user_b):
    client = anthropic.AsyncAnthropic()
    agents = [
        ("a", user_a, _system_prompt(user_a["display_name"], user_b["display_name"],
                                     _PERSONA_TEMPLATES[0], topic)),
        ("b", user_b, _system_prompt(user_b["display_name"], user_a["display_name"],
                                     _PERSONA_TEMPLATES[1], topic)),
    ]
    transcript = []
    try:
        for i in range(num_messages):
            key, user, system = agents[i % 2]
            resp = await client.messages.create(
                model=_MODEL,
                max_tokens=200,
                thinking={"type": "disabled"},
                system=system,
                messages=_build_messages(transcript, key, topic),
            )
            text = next((b.text for b in resp.content if b.type == "text"), "").strip()
            if not text:
                raise RuntimeError("Empty completion from model.")
            transcript.append((key, text))

            # Byte-identical to the WS handler event shape so the whole pipeline
            # (analysis, persistence, reasoning, broadcast) treats it as a real message.
            event = {
                "text":            text,
                "conversation_id": conv_id,
                "user_id":         user["user_id"],
                "timestamp":       time.time(),
                "message_id":      str(uuid.uuid4()),
                "metadata":        {"source": "websocket"},
            }
            await _redis_client.publish_event("message_stream", event)
            _state["sent"] = i + 1
            logger.bind(message_id=event["message_id"], conversation_id=conv_id,
                        user_id=user["user_id"]).info(
                f"event=ai_demo_message sent={i + 1}/{num_messages}")
            if i < num_messages - 1:
                await asyncio.sleep(_PACING_SECONDS)
        _state["status"] = "done"
        logger.info(f"event=ai_demo_done conversation_id={conv_id} total={num_messages}")
    except asyncio.CancelledError:
        _state["status"] = "stopped"
        logger.info(f"event=ai_demo_stopped conversation_id={conv_id} sent={_state['sent']}")
    except Exception as e:
        _state["status"] = "error"
        _state["error"]  = str(e)[:300]
        logger.log_exception("ai_demo_run_failed", e)


@router.post("")
async def start_ai_demo(req: StartAiDemoRequest, admin: dict = Depends(require_admin)):
    global _task
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured.")

    async with _lock:
        if _task and not _task.done():
            raise HTTPException(status_code=409, detail="An AI demo is already running.")

        pool = get_pool()
        async with pool.acquire() as conn:
            taken = set()
            user_a = await _create_agent_user(conn, taken)
            user_b = await _create_agent_user(conn, taken)

        res = await create_group_conversation(
            CreateGroupRequest(
                name=f"AI Demo: {req.topic}"[:100],
                member_ids=[user_a["user_id"], user_b["user_id"]],
            ),
            current_user=admin,
        )
        conv_id = res["conversation_id"]

        _state.update(status="running", conversation_id=conv_id, topic=req.topic,
                      sent=0, total=req.num_messages, error=None)
        _task = asyncio.create_task(
            _run_ai_demo(conv_id, req.topic, req.num_messages, user_a, user_b)
        )

    logger.info(f"event=ai_demo_started admin={admin['sub']} conversation_id={conv_id} "
                f"topic={req.topic!r} total={req.num_messages}")
    return {"conversation_id": conv_id, "status": "running", "total": req.num_messages}


@router.get("/status")
async def ai_demo_status(admin: dict = Depends(require_admin)):
    return dict(_state)


@router.post("/stop")
async def stop_ai_demo(admin: dict = Depends(require_admin)):
    if _task and not _task.done():
        _task.cancel()
        _state["status"] = "stopped"
        logger.info(f"event=ai_demo_stop_requested admin={admin['sub']}")
    return dict(_state)
