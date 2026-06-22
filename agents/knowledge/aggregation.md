# aggregation Knowledge Base
Last updated: 2026-06-21

## Architecture
Reads `emotion_stream`, maintains per-conversation state, publishes to `conversation_update_stream`.

**Per-speaker valence tracking** (key: `conv:{cid}:spk:{uid}:valence_seq`):
- JSON list, oldest-first, appended after each message
- Read by context_engine for velocity/acceleration calculation
- Written by aggregation_service after meta-learner prediction arrives

**Conversation hash** (`conversation:{cid}`):
- Per-speaker valence entries keyed by user_id
- Used by context_engine for speaker_divergence (std across speakers)

**Dynamics block** (feature vector [111:113]):
- [111] emotional_inertia — how much current emotion persists from previous (Kuppens)
- [112] contagion — cross-speaker emotional influence (Kramer)

**Aggregation flow** (in central_responder/main.py):
1. Accumulate partials in `pending_aggregations[message_id]`
2. All required models (vader, basic_bert, go_emotions) → start OPTIONAL_TIMEOUT_MS timer
3. Optional context_engine arrives → cancel timer, aggregate immediately
4. `aggregate_and_publish` builds 116-dim vector, runs meta-learner, publishes to emotion_stream
5. aggregation_service reads emotion_stream → updates conversation state → publishes to conversation_update_stream

**Late-arrival deduplication**: `_completed_ids` deque prevents re-processing.
**Stale cleanup**: messages with no aggregation after 60s are evicted with WARNING log.

## Known Issues
- None currently recorded.

## Improvement Queue
- **[Med]** Expose mood_trajectory arc (last N valence values) via WebSocket for frontend arc chart.
- **[Low]** Add aggregation_service Prometheus counters for messages processed / latency.

## Cross-Agent Dependencies
- Provides: `conv:{cid}:spk:{uid}:valence_seq` to **context_engine**
- Provides: `conversation:{cid}` hash (per-speaker valence) to **context_engine**
- Depends on: **pipeline** for emotion_stream messages
- Depends on: **infra** for Redis key namespacing
- Depends on: **meta_learner** for emotion predictions (reads emotion_stream output)

## Inter-Agent Requests (Pending)
*None*

## Recent History
- 2026-06-20: Added per-speaker valence_seq writing (was previously conversation-level only)
