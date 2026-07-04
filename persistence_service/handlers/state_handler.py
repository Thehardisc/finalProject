"""persistence_service/handlers/state_handler.py — Persist conversation state and escalation score."""
import json

from shared.utils.logger import get_logger
from persistence_service.db_models import EmotionAnalysis, Message, ConversationState

logger = get_logger("persistence_service")

TENSION_LABELS = ['anger', 'annoyance', 'sadness', 'grief', 'disgust', 'fear']


def calculate_sentiment_velocity(session, conversation_id: str, limit: int = 5) -> float:
    """Compute a scalar escalation score from the recent sentiment trend."""
    try:
        results = (
            session.query(EmotionAnalysis.emotions_json)
            .join(Message, Message.message_id == EmotionAnalysis.message_id)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp.desc())
            .limit(limit).all()
        )

        if len(results) < 2:
            return 0.0

        scores = []
        for (json_str,) in reversed(results):
            data    = json.loads(json_str)
            tension = sum(data.get(label, 0.0) for label in TENSION_LABELS)
            scores.append(tension)

        diffs       = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        avg_velocity = sum(diffs) / len(diffs) if diffs else 0.0
        final_score  = scores[-1] + (avg_velocity * 0.5)
        return max(0.0, min(1.0, final_score))

    except Exception as e:
        logger.warning(f"Velocity calc failure: {e}")
        return 0.0


async def process_state_event(session, data: dict) -> None:
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        logger.warning("No conversation_id provided in state event, skipping.")
        return

    state_json = data.get("conversation_state", "{}")
    try:
        state_obj = json.loads(state_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Malformed conversation_state JSON for {conversation_id}, using empty state.")
        state_obj = {}

    escalation_score = calculate_sentiment_velocity(session, conversation_id)

    state_record = (session.query(ConversationState)
                    .filter(ConversationState.conversation_id == conversation_id)
                    .first())
    if not state_record:
        state_record = ConversationState(conversation_id=conversation_id)
        session.add(state_record)

    state_record.state_json      = state_json
    state_record.escalation_score = float(escalation_score)
    state_record.last_updated     = float(state_obj.get("last_updated", 0))
