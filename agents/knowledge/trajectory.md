# trajectory Knowledge Base
Last updated: 2026-06-21

## Architecture
**EmotionalDialogueEncoder (EDE)** — LSTM-based trajectory model.
- File: `central_responder_service/trajectory/model.py`
- Inference: `central_responder_service/trajectory/inference.py`
- Model files: `central_responder_service/models/` (bind-mounted)

**Input**: GoEmotions history window `[B, N, 28]`
- N=12 (window size), d=64 (hidden dim)
- Oldest message first

**Output**:
- `predicted_next [B, 28]` — softmax over 28 GoEmotions classes
- `phase_logits [B, 6]` — conversation phase classification

**6 Conversation Phases**: opening, escalation, peak, turning_point, resolution, sustained

**Redis keys**:
- `trajectory:{conv_id}:prior` — 28-dim JSON list (predicted_next for next message)
- `trajectory:{conv_id}:phase` — current phase string

**Feedback loop**:
1. inference.py writes `predicted_next` → `trajectory:{conv_id}:prior`
2. central_responder/main.py reads prior BEFORE `build_feature_vector` call
3. Prior goes into feature slot [82:110] for message T+1
4. Zeros on first message or if model files missing (graceful degradation)

**input_dim**: read from config file in models/ directory (not hardcoded)

**LSTM Top-3 accuracy**: ~81% (as of last training run)

## Known Issues
- Prior defaults to zeros if model files missing — acceptable degradation but ideally logged at DEBUG.
- Phase written to Redis but not surfaced in WebSocket payload to frontend (only logged in pipeline_log).

## Improvement Queue
- **[High]** Expose conversation phase in WebSocket payload for frontend display — users should see "escalation" / "resolution" in real time.
- **[Med]** Add trajectory arc chart to frontend (predicted GoE distribution over next 3 messages).
- **[Low]** Phase transition detection: fire a special event when phase changes (e.g., escalation→resolution).

## Cross-Agent Dependencies
- Provides: 28-dim prior [82:110] to **meta_learner** via Redis
- Reads: GoEmotions output from **nlp** (via aggregation in central_responder)
- Depends on: **infra** for Redis `trajectory:*` key namespace
- Depends on: **pipeline** for message ordering (window must be chronological)

## Inter-Agent Requests (Pending)
*None*

## Recent History
- 2026-06-20: Trajectory LSTM top-3 accuracy 81% — stable
- 2026-06-20: Prior correctly initialized to zeros on first message of each conversation
