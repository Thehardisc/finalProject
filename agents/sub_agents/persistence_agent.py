from .base_agent import BaseAgent


class PersistenceAgent(BaseAgent):
    name = "persistence"
    description = (
        "PostgreSQL writer: batches emotion events from emotion_stream to DB. "
        "Dead-letter queue (DLQ) on write failure — dlq_write_failed log event when DLQ itself fails. "
        "Owns: persistence_service/. "
        "Key log events: persist_batch_done, persistence_failed, dlq_write_failed. "
        "Tables: messages, emotions, conversations, users."
    )
    key_files = [
        "persistence_service/main.py",
        "shared/constants.py",
    ]
