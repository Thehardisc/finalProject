ok# CLAUDE.md — InnerLink Emotion Analysis System

## Project Summary
InnerLink is a real-time emotion analysis chat application. Messages flow through an ML pipeline of parallel NLP analyzers, a meta-learner that fuses their outputs, and a trajectory LSTM that predicts emotional direction. The frontend displays live emotion scores alongside each message.

---

## Service Map

| Service | Port | Purpose |
|---|---|---|
| `ingestion_service` | 8000 | REST entry point — accepts messages, publishes to Redis `message_stream` |
| `preprocessing_service` | — | Normalizes text, publishes to `preprocessed_stream` |
| `vader_service` | — | Lexicon sentiment (4 scores) |
| `bert_service` | — | 7-class Ekman emotion via `j-hartmann/emotion-english-distilroberta-base` |
| `goemotions_service` | — | 28-class GoEmotions via `bhadresh-savani/bert-base-go-emotion` + emoji scoring |
| `central_responder_service` | — | Fuses all model outputs → meta-learner inference + background retraining |
| `aggregation_service` | — | Conversation state tracking (valence, mood trajectory) |
| `llm_reasoning_service` | — | Optional LLM reasoning layer (default: RULE_BASED) |
| `context_engine_service` | — | Episodic memory via Qdrant vector DB |
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
          → emotion_stream
              → conversation_update_stream
```

`goemotions_service` also computes emoji scores (via `emoji.demojize` → GoEmotions model) and includes them in its `partial_analysis_stream` event as `emoji_scores`.

`central_responder_service` reads the `emoji_scores` field from the go_emotions result and injects it as `model_outputs["emojinet"]` before building the feature vector.

---

## Feature Vector (118 dimensions)

**Three** builders MUST produce the same layout: `central_responder_service/meta_learner.py:build_feature_vector` (live path, called by `main.py`), `central_responder_service/ml/predictor.py:build_feature_vector` (modular path), and `central_responder_service/trainer/preprocessor.py:build_fv` (training). Any change to one must be mirrored in the others. The CDM context + derived blocks are built by shared helpers in `ml/features.py` so they cannot silently desync.

| Block | Indices | Source | Size |
|---|---|---|---|
| VADER | [0:4] | `vader_neg, vader_neu, vader_pos, vader_compound` | 4 |
| BERT Ekman | [4:11] | 7 Ekman labels from `BERT_LABELS` | 7 |
| GoEmotions | [11:39] | 28 labels from `EMOTION_LABELS` | 28 |
| EmojiNet | [39:67] | 28 emoji-derived scores (same label order as GoEmotions) | 28 |
| CDM Context | [67:111] | 28-dim PBSM belief + 16 CDM scalars (see below) | 44 |
| Derived | [111:118] | bert_entropy, goe_entropy, bert_margin, goe_margin, bert_goe_agreement, vader_abs_compound, max_goe_score | 7 |

The **CDM Context block (44)** is produced by `context_engine_service`'s Parallel Belief State Machine (`context_engine_service/cdm.py`) and published on `partial_analysis_stream` as `context_vector`. `central_responder_service/main.py` injects it as `context["cdm_context"]`. Its internal layout (relative to the block) is defined by the `CTX_*` constants in `shared/constants.py`: `[0:28]` belief distribution, `[28]` residency, `[29:32]` transition trace, `[32:44]` scalars (abruptness, coherence, entropy, speaker-divergence, velocity, acceleration, historical/current valence, resonance, volatility, msg-length, latency). When the context engine misses the aggregation window, the block falls back to a **uniform belief** (1/28 each, scalars zeroed) — never silently zeroed. The trainer synthesises a soft **Dirichlet** belief peaked at the previous emotion to mirror the live distribution's shape (avoids train/serve skew).

Label lists live in `shared/constants.py`: `EMOTION_LABELS` (28), `BERT_LABELS` (7), `VADER_KEYS` (4).

---

## Trajectory LSTM

- **Input per step**: 67-dim tensor — GoEmotions(28) + BERT(7) + VADER(4) + EmojiNet(28)
- **Built in**: `central_responder_service/trajectory/inference.py:build_feature_vector`
- **Model files**: `central_responder_service/models/trajectory_lstm.pt` + `trajectory_config.json`
- **Purpose**: predicts the emotional direction of the next message in a conversation
- The model is optional — inference degrades gracefully if the file is missing

---

## Meta-Learner Retraining

The trainer runs as a background thread inside `central_responder_service`.

- **Entry**: `central_responder_service/trainer/runner.py:start_trainer_thread`
- **Feature building**: `trainer/preprocessor.py:build_fv` (must match inference exactly)
- **Analyzers**: `trainer/analyzers.py` — loads VADER, BERT, GoEmotions transiently, then unloads
- **Data**: fetched from PostgreSQL via `trainer/data_fetcher.py`
- **Output**: `central_responder_service/models/meta_weights.pkl` (+ `.sha256`, `_meta.json`, `.ready`)
- **Gate**: new model only hot-reloaded if test accuracy ≥ `ACCURACY_GATE` (default 0.40)
- **Interval**: `RETRAIN_INTERVAL_SECONDS` (default 1800)

Key invariant: `trainer/preprocessor.py:build_fv` and `ml/predictor.py:build_feature_vector` must produce identical feature vectors. A mismatch causes `X has N features but StandardScaler expecting M` errors.

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

- **Feature vector parity**: `meta_learner.py`, `ml/predictor.py`, and `trainer/preprocessor.py` must produce the same 118-dim vector. Whenever you change one, update the others (the shared `ml/features.py` helpers cover the CDM-context and derived blocks). Run `pytest central_responder_service/training/test_feature_parity.py`.
- **Trajectory input**: always 67-dim (GoE+BERT+VADER+Emoji). The trajectory model was trained with this layout — changing it requires retraining the LSTM.
- **Emoji scores flow**: `goemotions_service` computes emoji scores and attaches them to its result. `central_responder_service/main.py` reads `emoji_scores` from the go_emotions result and puts it into `model_outputs["emojinet"]` before calling `build_feature_vector`.
- **CSS import**: `main.jsx` imports `index-v2.css`, not `index.css`. Edits to `index.css` have no effect on the running app.
- **Model files on host**: `central_responder_service/models/` is bind-mounted into the container. Model files saved by the trainer are immediately visible to the running service without a rebuild.
- **Frontend rebuild required**: CSS and JSX changes require `docker compose up --build frontend_service -d` — the container serves a static Vite build.
- **`trainer/` package**: `central_responder_service/trainer/__init__.py` must exist so Python picks the `trainer/` package over the legacy `trainer.py` monolith file.

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

The bind helper is composable — add per-block fields without losing the base context:

```python
mlog = logger.bind(message_id=mid, conversation_id=cid)
mlog.info("done", extra={"event": "model_done", "latency_ms": elapsed})
```

### Log level contract

| Level | Use for |
|---|---|
| `DEBUG` | Dev-only verbosity (per-message dispatch lines, intermediate computations). Off in prod. |
| `INFO` | Normal traffic + structured audit events (`user_login`, `meta_inference`, etc). |
| `WARNING` | Recoverable degradation (parse failures, fallback paths, rate limits, NOGROUP recovery). |
| `ERROR` | Data loss, repeated retry exhaustion, transaction rollback, unhandled exceptions. |
| `CRITICAL` | Pageable — service can no longer make forward progress. |

### Standard `event=` field values

These are the structured event names used across the stack. Grep one to find every occurrence:

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
| `persist_batch_done` / `persistence_failed` | persistence_service | DB batch outcome (failed includes `message_ids` for replay) |
| `dlq_write_failed` | persistence_service | A DLQ entry itself couldn't be written |
| `user_registered` / `user_login` | api_service auth | Audit success events; `email_hash` only, never raw email |
| `login_failed` (`reason=user_not_found\|account_inactive\|password_mismatch`) | api_service auth | Audit failure events |
| `ws_auth_failed` (`reason=missing_cookie\|user_mismatch\|decode_error`) | api_service WS | Audit failure events |
| `admin_action` (`action=update_user\|delete_user`) | api_service admin | Records `actor` + `target` |
| `trainer_phase` (`phase=data_load\|feature_extract_*\|live_aug\|filter\|train\|eval`) | trainer | Each phase end with `duration_ms` |
| `trainer_gate_decision` | trainer | `test_acc`, `threshold`, `delta`, `deployed` |
| `trainer_cycle_complete` | trainer | Total duration + deployed bool |
| `trainer_heartbeat` | trainer | Once per minute between cycles — proves the loop is alive |
| `model_hot_reload` | central_responder | Logs the prev_acc → new_acc transition |
| `emoji_scores_parse_failed` | central_responder | Structured fields only — raw bytes never logged |

### Log rotation

Every service in `docker-compose.yml` uses the `x-log-rotation` YAML anchor: `json-file` driver, `max-size: 10m`, `max-file: 3`. Bounded ~30 MB per container in any long-running deploy.
