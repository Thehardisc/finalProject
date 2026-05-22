"""
persistence_service/handlers/message_handler.py — Persist incoming chat messages.
"""
from persistence_service.db_models import Message


async def process_message_event(session, data: dict) -> None:
    msg = Message(
        message_id=      data.get("message_id"),
        conversation_id= data.get("conversation_id"),
        user_id=         data.get("user_id"),
        text=            data.get("text"),
        timestamp=       float(data.get("timestamp", 0))
    )
    session.merge(msg)
