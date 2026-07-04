"""persistence_service/handlers/emotion_handler.py — Persist emotion analysis results."""
from persistence_service.db_models import EmotionAnalysis


async def process_emotion_event(session, data: dict) -> None:
    msg_id = data.get("message_id")
    existing = session.query(EmotionAnalysis).filter_by(message_id=msg_id).first()
    if existing:
        existing.emotions_json     = data.get("emotions", "{}")
        existing.reasoning_json    = data.get("reasoning")
        existing.pipeline_log_json = data.get("pipeline_log")
    else:
        session.add(EmotionAnalysis(
            message_id=       msg_id,
            emotions_json=    data.get("emotions", "{}"),
            reasoning_json=   data.get("reasoning"),
            pipeline_log_json=data.get("pipeline_log"),
        ))
