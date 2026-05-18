"""
persistence_service/handlers/emotion_handler.py — Persist emotion analysis results.
"""
from persistence_service.db_models import EmotionAnalysis


async def process_emotion_event(session, data: dict) -> None:
    analysis = EmotionAnalysis(
        message_id=       data.get("message_id"),
        emotions_json=    data.get("emotions", "{}"),
        reasoning_json=   data.get("reasoning"),
        pipeline_log_json=data.get("pipeline_log")
    )
    session.add(analysis)
