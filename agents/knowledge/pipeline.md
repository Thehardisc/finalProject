# pipeline Knowledge Base
Last updated: 2026-06-21

## Architecture
Redis Stream topology (left → right):
```
message_stream
  → preprocessed_stream
      → partial_analysis_stream   (vader, basic_bert, go_emotions, context_engine — parallel)
          → emotion_stream
              → conversation_update_stream
```

- **ingestion_service** (port 8000): REST entry. Writes to `message_stream`.
- **preprocessing_service**: debouncer + text normalization, reads `message_stream`, writes `preprocessed_stream`.
- Each NLP service reads `preprocessed_stream`, writes to `partial_analysis_stream` with `model_name` field.
- `context_engine_service` is OPTIONAL (module_registry). If it misses `OPTIONAL_TIMEOUT_MS`, aggregation proceeds with zero ctx vector.
- `central_responder_service` accumulates partials in `pending_aggregations[message_id]`, fires after all required models arrive.
- After aggregation: publishes to `emotion_stream`, then `aggregation_service` writes to `conversation_update_stream`.

Key env vars: `OPTIONAL_TIMEOUT_MS=1000`, `INTERNAL_API_KEY` (X-API-Key header on all endpoints).

Consumer groups: each service uses XREADGROUP with its own group name. On NOGROUP error → XGROUP CREATE with `$` then retry.

## Known Issues
- None currently recorded.

## Improvement Queue
- **[Med]** Add dead-letter queue for messages that stale past 60s eviction window — currently just logged as WARNING.
- **[Low]** Pipeline observability: Prometheus counter for each stream hop latency.

## Cross-Agent Dependencies
- Provides: raw message flow to **nlp**, **context_engine**, **trajectory**, **meta_learner**
- Depends on: **infra** for Redis config, stream creation, NOGROUP recovery pattern
- Depends on: **meta_learner** for aggregation logic (pending_aggregations, _completed_ids deque)

## Inter-Agent Requests (Pending)
*None*

## Recent History
- 2026-06-20: start.sh indent bug fixed (indentation error in bash script caused silent exit)
- 2026-06-20: per-speaker context isolation added — aggregation_service now writes `conv:{cid}:spk:{uid}:valence_seq`
