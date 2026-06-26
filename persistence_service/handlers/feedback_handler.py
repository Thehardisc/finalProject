"""
persistence_service/handlers/feedback_handler.py — Persist human-verified emotion labels.
"""
from shared.utils.logger import get_logger
from persistence_service.db_models import EmotionAnalysis

logger = get_logger("persistence_service")


async def process_feedback_event(session, data: dict) -> None:
    msg_id = data.get("message_id")
    label  = data.get("ground_truth_emotion")

    if not msg_id or not label:
        logger.warning(f"Feedback event dropped — missing field(s): message_id={msg_id!r}, label={label!r}")
        return

    analysis = (session.query(EmotionAnalysis)
                .filter(EmotionAnalysis.message_id == msg_id).first())
    if analysis:
        analysis.ground_truth_emotion = label
        analysis.is_verified          = True
        logger.info(f"Feedback applied to {msg_id}: Label={label}")
    else:
        new_analysis = EmotionAnalysis(
            message_id=           msg_id,
            ground_truth_emotion= label,
            is_verified=          True,
            emotions_json=        "{}"
        )
        session.add(new_analysis)
        logger.info(f"New verified analysis created for {msg_id}: Label={label}")
