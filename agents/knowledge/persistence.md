# persistence Knowledge Base
Last updated: 2026-06-21

## Architecture
Batches emotion events from `emotion_stream` into PostgreSQL.

**Tables**:
- `messages` — raw message text + metadata
- `emotions` — predicted emotion labels + scores per message
- `conversations` — conversation metadata
- `users` — user accounts (password hash, JWT, role)

**Dead-letter queue (DLQ)**:
- On write failure: appends to DLQ file/table
- On DLQ write failure: logs `dlq_write_failed` at ERROR level

**Key log events**:
- `persist_batch_done` — INFO, batch written successfully
- `persistence_failed` — ERROR, write failed (triggers DLQ)
- `dlq_write_failed` — ERROR, DLQ itself failed (data at risk)

## Known Issues
- None currently recorded.

## Improvement Queue
- **[Med]** DLQ retry mechanism — currently DLQ is write-only, no automatic replay.
- **[Low]** Add row count to persist_batch_done log for observability.

## Cross-Agent Dependencies
- Reads: emotion_stream from **meta_learner** (via central_responder)
- Depends on: **infra** for PostgreSQL connection + table schema

## Inter-Agent Requests (Pending)
*None*

## Recent History
*No incidents recorded.*
