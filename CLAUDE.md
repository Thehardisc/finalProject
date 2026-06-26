# CLAUDE.md — InnerLink Emotion Analysis System

---

## Workflow Protocol

### Philosophy: The 80/20 Rule
Use AI for the 80% routine/heavy-lifting tasks; surface the remaining 20% (judgment calls) explicitly for human review.

### Codification Pipeline
For every complex or repetitive task, follow these 4 steps:

1. **[PIPELINE DEFINITION]** — Outline inputs, processing logic, and expected output before starting.
2. **[EXECUTION OF THE 80%]** — Do the heavy lifting: formatting, drafting, data extraction, initial implementation.
3. **[THE 20% JUDGMENT CALL]** — Stop and explicitly flag areas requiring human judgment, strategic choice, or approval. Use clear formatting.
4. **[HUMAN-IN-THE-LOOP]** — Wait for approval or modification on the flagged 20% before finalizing.

### Operational Rules
- **High Signal, Low Noise**: No pleasantries. Output tables, diff blocks, and structured lists.
- **Identify Edge Cases**: Do not guess on ambiguities — quarantine and flag them in the [THE 20% JUDGMENT CALL] section.
- **Repeatable Structure**: Treat requests as reusable blueprints with consistent output format every time.

---

## Project Summary
InnerLink is a real-time emotion analysis chat application. Messages flow through an ML pipeline of parallel NLP analyzers, a meta-learner that fuses their outputs, and a trajectory LSTM that predicts emotional direction. The frontend displays live emotion scores alongside each message.

---

## Service Map

| Service | Port | Purpose |
|---|---|---|
| `ingestion_service` | 8000 | REST entry point — accepts messages, publishes to Redis `message_stream` |
| `preprocessing_service` | — | Normalizes text (with debouncer), publishes to `preprocessed_stream` |
| `vader_service` | — | Lexicon sentiment (4 scores) |
| `bert_service` | — | 7-class Ekman emotion via `j-hartmann/emotion-english-distilroberta-base` |
| `goemotions_service` | — | 28-class GoEmotions via `bhadresh-savani/bert-base-go-emotion` + emoji scoring |
| `central_responder_service` | — | Fuses all model outputs → meta-learner inference + background retraining |
| `aggregation_service` | — | Conversation state tracking (valence, mood trajectory) |
| `llm_reasoning_service` | — | Optional LLM reasoning layer (default: RULE_BASED) |
| `context_engine_service` | — | CDM (Conversation Dynamics Machine) + episodic memory via Qdrant |
| `persistence_service` | — | Writes to PostgreSQL |
| `api_service` | 8001 | WebSocket + REST API for frontend |
| `frontend_service` | 5173 | React app served via Nginx |
| `qdrant` | 6333 | Vector DB for context engine |
| `redis` | 6379 | Message bus (Streams) + state (Hashes) |
| `db` (postgres) | 5432 | Long-term storage |

---

## Redis Stream Pipeline

```
message_stream
  → preprocessed_stream
      → partial_analysis_stream   (vader, basic_bert, go_emotions — all parallel)
                                  (context_engine — optional, CDM vector)
          → emotion_stream
              → conversation_update_stream
```

`goemotions_service` computes emoji scores (via `emoji.demojize` → GoEmotions model) and includes them in its `partial_analysis_stream` event as `emoji_scores`. These are forwarded but not currently incorporated into the meta-learner feature vector.

`context_engine_service` publishes a 40-dim CDM context vector (`CDM_CTX_DIM`) as `model_name=context_engine`. The central responder waits for all required models, then optionally uses the context vector if it arrived within `OPTIONAL_TIMEOUT_MS`.

`trajectory/inference.py` saves the LSTM's `predicted_next` (28-dim GoEmotions distribution) to Redis key `trajectory:{conv_id}:prior` after each step. `aggregate_and_publish` reads this prior before calling `build_feature_vector` so the next message gets the trajectory prediction as an additional input block.

---

## Feature Vector (116 dimensions)

`central_responder_service/meta_learner.py:build_feature_vector` (inference) and `central_responder_service/trainer/` (training) MUST produce the same layout.

| Block | Indices | Source | Size | Constant |
|---|---|---|---|---|
| VADER | [0:4] | `vader_neg, vader_neu, vader_pos, vader_compound` | 4 | — |
| BERT Ekman | [4:11] | 7 Ekman labels from `BERT_LABELS` | 7 | — |
| GoEmotions | [11:39] | 28 labels from `EMOTION_LABELS` | 28 | — |
| VAD lexicon | [39:42] | valence, arousal, dominance (Warriner 2013) | 3 | `VAD_DIM=3` |
| CDM Context | [42:82] | context_engine CDM+HMM+scalars | 40 | `CDM_CTX_DIM=40` |
| Trajectory Prior | [82:110] | EDE `predicted_next` from previous message | 28 | `PRIOR_DIM=28` |
| Sarcasm | [110] | sarcasm classifier score [0,1] | 1 | `SARCASM_DIM=1` |
| Dynamics | [111:113] | emotional inertia + contagion (Kuppens/Kramer) | 2 | `DYNAMICS_DIM=2` |
| Appraisal | [113:116] | novelty, goal_congruence, coping (Scherer 2001) | 3 | `APPRAISAL_DIM=3` |

`ML_DIM=42` (NLP block). `CONTEXT_DIM = CDM_CTX_DIM + PRIOR_DIM + SARCASM_DIM + DYNAMICS_DIM + APPRAISAL_DIM = 74` — this is the full context block that `GatingEnsembleNet` processes as `x_c = x[:, ML_DIM:FEATURE_DIM]`.

**CDM context vector layout** (40 dims — indices are *within* the CDM block, i.e. offset from 39):
- `[0:15]` CDM intent one-hot (15 states — `N_CDM_STATES=15`)
- `[15]` state_residency
- `[16:19]` transition_path (last 3 state indices / N_CDM_STATES)
- `[19]` entry_abruptness
- `[20]` topic_coherence
- `[21]` emotion_entropy
- `[22]` speaker_divergence
- `[23]` velocity (Δvalence)
- `[24]` acceleration (Δ²valence)
- `[25]` hist_valence_pos (Qdrant, positive component)
- `[26]` hist_valence_neu (Qdrant, neutral component)
- `[27]` hist_valence_neg (Qdrant, negative component)
- `[28]` topic_resonance
- `[29]` volatility
- `[30]` current_valence
- `[31]` message_length
- `[32]` latency_ms
- `[33]` hmm_conf (`CTX_HMM_CONF` — drives ctx_weight in GatingEnsembleNet)
- `[34]` hmm_entropy
- `[35]` hmm_emission_logprob
- `[36:39]` hmm_top3_next_probs
- `[39]` intent_stability

**Trajectory Prior block** (28 dims, [82:110]): GoEmotions distribution predicted by the EDE for this message, fetched from Redis `trajectory:{conv_id}:prior`. Zeros on first message or when trajectory model is absent.

Label lists live in `shared/constants.py`: `EMOTION_LABELS` (28), `BERT_LABELS` (7), `VADER_KEYS` (4), `CDM_STATES` (15).
Named index constants (`CTX_*`) must be used instead of hardcoded integers.

---

## Trajectory — EmotionalDialogueEncoder (EDE)

- **Architecture**: `central_responder_service/trajectory/model.py:EmotionalDialogueEncoder`
- **Input**: GoE history window `[B, N, 28]` — last N real GoE distributions (oldest first). `d=64`, `window=12`.
- **Output**: prior `[B, 28]` softmax + phase_logits `[B, 6]` (conversation phase)
- **Phases**: opening, escalation, peak, turning_point, resolution, sustained
- **Inference**: `central_responder_service/trajectory/inference.py`
- **Model files**: `central_responder_service/models/` — bind-mounted, visible without rebuild
- **Trajectory prior feedback**: after each step, `inference.py` writes `predicted_next` to Redis `trajectory:{conv_id}:prior`. Read before `build_feature_vector` → feature slot [82:110] for message T+1.
- **Phase**: written to `trajectory:{conv_id}:phase`. Logged in pipeline_log.
- The model is optional — inference degrades gracefully if files missing; prior defaults to zeros.

---

## Meta-Learner Retraining

The trainer runs as a background thread inside `central_responder_service`.

- **Entry**: `central_responder_service/trainer.py:start_trainer_thread`
- **Feature building**: `trainer.py` (must match inference in `meta_learner.py` exactly)
- **Data**: fetched from PostgreSQL
- **Output**: `central_responder_service/models/meta_weights.pkl` (+ `.sha256`, `_meta.json`, `.ready`)
- **Gate**: new model only hot-reloaded if test accuracy ≥ `ACCURACY_GATE` (default 0.40)
- **Interval**: `RETRAIN_INTERVAL_SECONDS` (default 1800)

Key invariant: `trainer/cycle.py` feature building and `meta_learner.py:build_feature_vector` must produce identical 116-dim vectors. A mismatch causes `X has N features but StandardScaler expecting M` errors.

---

## Central Responder Internals

`main.py` bootstraps the meta-learner, wires hot-reload, starts the background trainer + Prometheus metrics server (:9090), loads the trajectory LSTM, and runs the Redis consumer loop inline.

- **`meta_learner.py`** — all ML logic: `build_feature_vector`, `predict_with_meta_learner`, `calculate_feature_impacts`, `apply_context_correction`, `load_meta_learner`.
- **`trainer.py`** — background retraining daemon: `start_trainer_thread`.
- **`trajectory/inference.py`** — trajectory LSTM step; reads `input_dim` from config file.
- **`ml/conflict_detector.py`**, **`ml/loader.py`** — model loading + conflict detection helpers.
- **`shared/module_registry.py`** — dynamic registry of required/optional stream modules; enables the context engine to be optional without code changes.

**Aggregation flow**:
1. Accumulate partial results in `pending_aggregations` dict keyed by `message_id`
2. Once all required models arrive, start `OPTIONAL_TIMEOUT_MS` timer for optional modules
3. If optional (context_engine) arrives in time, cancel timer and aggregate immediately
4. `aggregate_and_publish` builds feature vector, runs meta-learner, publishes to `emotion_stream`

**Reliability**:
- **Rule-based fallback**: `predict_with_meta_learner` falls back to GoEmotions→BERT→VADER argmax when no model loaded
- **Late-arrival deduplication**: `_completed_ids` deque prevents re-processing after aggregation completes
- **Stale cleanup**: messages with no aggregation after 60s are evicted with a warning log

---

## Auth & Users

- JWT-based auth in `api_service/auth_utils.py`
- Registration creates a user in PostgreSQL; the username matching `ADMIN_USERNAME` env var gets admin role
- Routes: `api_service/routes/auth.py`, `users.py`, `conversations.py`, `messages.py`, `analytics.py`, `admin.py`
- `JWT_SECRET`, `JWT_EXPIRY_HOURS`, `ADMIN_USERNAME` configured in `.env`

---

## Frontend

- **Framework**: React (Vite build, served by Nginx in Docker)
- **Entry point**: `frontend_service/src/main.jsx` — imports `index-v2.css` (NOT `index.css`)
- **CSS**: `index-v2.css` (base styles) + `glass/CrystalGlass-v2.css` (design system, `.crystal-shell`)
- **Main app shell**: `App.jsx` wraps everything in `<div className="crystal-shell">`
- **Main view**: `src/pages/IGDashboard.jsx` — Instagram-style layout with sidebar + chat area
- **Scroll rule**: body and `.crystal-shell` are `overflow: hidden`. Only the messages container (`ref={messagesContainerRef}`) scrolls. Use `el.scrollTop = el.scrollHeight` to scroll to bottom — do NOT use `scrollIntoView` (it picks the wrong scroll ancestor).
- **Flex layout**: chat area needs `minHeight: 0` to constrain the flex column; messages div needs `minHeight: 0` for `overflowY: auto` to activate.

---

## Environment (.env)

Key variables (all in `.env`, applied to containers via `docker-compose.yml`):

```
TZ=Asia/Jerusalem               # Container timezone for log timestamps
INTERNAL_API_KEY=...            # X-API-Key header required on all endpoints
JWT_SECRET=...                  # Change in production
REDIS_PASSWORD=                 # Leave empty for no auth (dev)
RETRAIN_INTERVAL_SECONDS=1800
ACCURACY_GATE=0.40
MAX_SAMPLES=2500
LLM_PROVIDER=RULE_BASED         # or OPENAI, GROQ
ADMIN_USERNAME=admin
OPTIONAL_TIMEOUT_MS=1000        # ms to wait for context_engine after required models arrive
```

---

## Common Commands

```bash
# Start everything
docker compose up --build -d

# Rebuild a single service after code changes
docker compose up --build <service_name> -d

# Watch logs for a service
docker logs -f projects-final-central_responder_service-1

# Check all running services
docker compose ps
```

---

## Key Invariants & Gotchas

- **Feature vector parity**: `meta_learner.py:build_feature_vector` and `trainer/cycle.py` must produce the same 116-dim vector. Mismatch → `X has N features but StandardScaler expecting M` error.
- **Dimension constants**: `shared/constants.py` is the single source of truth. Key values: `ML_DIM=42`, `CDM_CTX_DIM=40`, `PRIOR_DIM=28`, `CONTEXT_DIM=74`, `FEATURE_DIM=116`, `N_CDM_STATES=15`. Use `CTX_*` named index constants — never hardcode integers. `context_engine_service` uses `CDM_CTX_DIM`; `GatingEnsembleNet` uses `CONTEXT_DIM` (the full 74-dim block).
- **Trajectory prior Redis key**: `trajectory:{conv_id}:prior` — 28-dim JSON list, feature slot [82:110]. Written by `trajectory/inference.py`, read by `aggregate_and_publish` in `central_responder/main.py`. Missing key → zeros (safe).
- **GoEmotions gate cap**: `GatingEnsembleNet.forward()` hard-caps GoEmotions gate weight at ≤0.50 via `.clamp(max=0.50)`. Enforced regardless of training data distribution.
- **Gate weights format**: `gate_weights_alpha` in WebSocket payload is [vader, bert, goe, vad, ctx] (5 elements). UI displays only first 3.
- **CSS import**: `main.jsx` imports `index-v2.css`, not `index.css`. Edits to `index.css` have no effect on the running app.
- **Model files on host**: `central_responder_service/models/` is bind-mounted into the container. Model files saved by the trainer are immediately visible to the running service without a rebuild.
- **Frontend rebuild required**: CSS and JSX changes require `docker compose up --build frontend_service -d` — the container serves a static Vite build.
- **Absolute imports in central_responder**: `main.py` runs as a top-level script (CWD = service dir), so imports use absolute paths rooted at the service dir (`from meta_learner import ...`, `from trainer import ...`). Relative imports break under `python main.py`.
- **Module registry**: `shared/module_registry.py` determines which models are "required" vs "optional". Context engine is optional — if it doesn't arrive within `OPTIONAL_TIMEOUT_MS`, aggregation proceeds with a zero context vector.

---

## Logging

All services log via `shared/utils/logger.py:get_logger(name)`. Two env vars control format:

- **`LOG_LEVEL`**: `DEBUG | INFO | WARNING | ERROR | CRITICAL` — default `INFO`.
- **`LOG_FORMAT`**: `TEXT | JSON` — default `TEXT`. Invalid values fall back to `TEXT` with a stderr warning.

Both are set on every service via `docker-compose.yml` and can be overridden per-service in `.env`.

### Correlation IDs

Every per-message handler binds `message_id`, `conversation_id`, and `user_id` once at the top of its loop, so every subsequent log line carries those fields. Follow a single message across the stack with:

```bash
docker compose logs | grep <message_id>
```

In **TEXT** mode the fields render as a trailing `[message_id=... conversation_id=... user_id=...]` suffix. In **JSON** mode they're top-level keys (parseable with `jq`).

### Log level contract

| Level | Use for |
|---|---|
| `DEBUG` | Dev-only verbosity (per-message dispatch lines, intermediate computations). Off in prod. |
| `INFO` | Normal traffic + structured audit events (`user_login`, `meta_inference`, etc). |
| `WARNING` | Recoverable degradation (parse failures, fallback paths, rate limits, NOGROUP recovery). |
| `ERROR` | Data loss, repeated retry exhaustion, transaction rollback, unhandled exceptions. |
| `CRITICAL` | Pageable — service can no longer make forward progress. |

### Standard `event=` field values

| Event | Emitter | Meaning |
|---|---|---|
| `ingest_accepted` | ingestion | Message written to `message_stream` |
| `preprocess_done` | preprocessing | Text normalized and republished |
| `model_done` (with `model=`) | vader/bert/goemotions | Per-model inference complete |
| `model_failed` | vader/bert/goemotions | Per-model inference threw |
| `aggregate_start` / `meta_inference` | central_responder | Per-message aggregation + prediction |
| `aggregation_done` | aggregation_service | Conversation state updated |
| `ws_message_received` | api_service | Client sent a message over WebSocket |
| `ws_broadcast` | api_service | Emotion event broadcast to N recipients |
| `persist_batch_done` / `persistence_failed` | persistence_service | DB batch outcome |
| `dlq_write_failed` | persistence_service | A DLQ entry itself couldn't be written |
| `user_registered` / `user_login` | api_service auth | Audit success events; `email_hash` only, never raw email |
| `login_failed` | api_service auth | Audit failure events |
| `ws_auth_failed` | api_service WS | Audit failure events |
| `admin_action` | api_service admin | Records `actor` + `target` |
| `model_hot_reload` | central_responder | Logs the prev_acc → new_acc transition |
| `meta_inference` (with `decision_mode=meta-learner\|rule-based`) | central_responder | Per-message prediction |

### Log rotation

Every service in `docker-compose.yml` uses the `x-log-rotation` YAML anchor: `json-file` driver, `max-size: 10m`, `max-file: 3`. Bounded ~30 MB per container in any long-running deploy.
