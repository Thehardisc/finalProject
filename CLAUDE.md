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
| `goemotions_service` | — | 28-class GoEmotions — **3-model blend**: SamLowe/finetuned (0.55) + bhadresh-savani (0.25) + semantic-NLI MiniLM (0.20); reads `processed_text_demojized` |
| `central_responder_service` | — | Fuses all model outputs → meta-learner inference; loads trajectory EDE + sarcasm classifier |
| `trainer_service` | — | Background meta-learner retraining (`python -m trainer`); writes new `meta_weights.pkl` and signals reload via Redis pub/sub |
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

Emoji handling lives in **preprocessing**: `demojize_text` converts emojis to words in `processed_text_demojized` (raw `text` is preserved so VADER scores emoji directly). `goemotions_service` reads `processed_text_demojized` and blends three signals — primary (SamLowe or the finetuned `samlowe_finetuned`), secondary (bhadresh-savani), and a semantic-NLI cosine match (all-MiniLM-L6-v2 against 28 prototype sentences): `0.55·primary + 0.25·secondary + 0.20·semantic`. It does **not** emit a separate `emoji_scores` field.

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

The trainer is a **package** (`central_responder_service/trainer/`) and runs in one of two modes, selected by `TRAINER_EXTERNAL`:

- **External (default, `TRAINER_EXTERNAL=true`)**: the separate `trainer_service` container runs `python -m trainer` (→ `trainer/__main__.py` → `trainer/cycle.py:run_one_cycle`). On a new accepted model it publishes to Redis channel `model_reload_signal` (`RELOAD_CHANNEL`); `central_responder_service.main` subscribes and hot-reloads.
- **In-process (`TRAINER_EXTERNAL=false`)**: `trainer.start_trainer_thread(on_model_reload)` runs the same cycle as a background thread inside `central_responder_service`.

- **Feature building**: `trainer/cycle.py` (must match inference in `meta_learner.py` exactly).
- **Data**: PostgreSQL message history + cached open datasets (EmpatheticDialogues, MELD, GoEmotions) under `.cache/`.
- **Output**: `meta_weights.pkl` (+ `_meta.json`, `.ready`) written to `/app/models` (host `./.cache/`).
- **Gate**: new model only hot-reloaded if test accuracy ≥ `ACCURACY_GATE` (default 0.40).
- **Interval**: `RETRAIN_INTERVAL_SECONDS` (default 1800).
- **GoEmotions fine-tune**: `trainer/finetune_goe.py` can produce `samlowe_finetuned` (picked up as the GoEmotions primary when present).

Key invariant: `trainer/cycle.py` feature building and `meta_learner.py:build_feature_vector` must produce identical 116-dim vectors. A mismatch causes `X has N features but StandardScaler expecting M` errors.

---

## Central Responder Internals

`main.py` bootstraps the meta-learner, wires hot-reload (subscribes to `model_reload_signal` when `TRAINER_EXTERNAL=true`, else starts the in-process trainer thread), starts the Prometheus metrics server (:9090), loads the trajectory EDE + sarcasm classifier, and runs the Redis consumer loop inline.

- **`meta_learner.py`** — all ML logic: `build_feature_vector`, `predict_with_meta_learner`, `calculate_feature_impacts`, `apply_context_correction`, `load_meta_learner`, `detect_emotional_conflicts`, `GatingNetworkWrapper` (the former `ml/` helper package was inlined here).
- **`trainer/`** (package) — retraining: `cycle.py:run_one_cycle`, `__main__.py` (container entry), `finetune_goe.py`, `models.py`, `utils.py`. `start_trainer_thread` exposed for in-process mode.
- **`trajectory/inference.py`** + **`trajectory/model.py`** — EmotionalDialogueEncoder (EDE) step; reads `input_dim`/window from config.
- **`sarcasm_classifier.py`** — DistilBERT sarcasm/passive-aggression head (config `sarcasm_clf_config.json`); feature slot [110].
- **`implicit_emotion.py`** — implicit-emotion routing (`is_implicit_candidate`, `should_override`) for understated/indirect inputs.
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
- **Entry point**: `frontend_service/src/main.jsx` — imports, in order, `styles/tailwind.css`, `styles/design-system.css`, `styles/emotionBubbles.css`, then `index-v2.css`
- **Tailwind v4**: enabled via the `@tailwindcss/vite` plugin (`vite.config.js`); `styles/tailwind.css` is the entry — `@theme` design tokens + a `dark:` variant keyed to the `data-ig-theme` attribute. Preflight (global reset) is intentionally deferred to the end of the migration (the app is still heavily inline-styled).
- **CSS**: `index-v2.css` + `styles/design-system.css` + `styles/emotionBubbles.css` (base styles, via `main.jsx`); `App.jsx` separately imports `glass/CrystalGlass-v2.css` (design system, provides `.crystal-shell`)
- **Chat theming**: `src/pages/IGDashboard.jsx` imports `styles/ig-theme.css` — light/dark tokens scoped to the chat subtree via the `data-ig-theme` attribute
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
TRAINER_EXTERNAL=true           # true → trainer_service container; false → in-process thread
MAX_EMPATHETIC_SAMPLES=25000    # cap on EmpatheticDialogues rows pulled into training
MIN_DB_SAMPLES=50               # min PostgreSQL messages before a retrain cycle runs
MODEL_PATH=/app/models/meta_weights.pkl
LLM_PROVIDER=RULE_BASED         # or OPENAI, GROQ
ANTHROPIC_API_KEY=...           # optional — agents/ CLI and synthetic data generation
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

# QA suite (offline = no stack; full incl. live needs the stack + .env)
./qa_suite/run_qa.sh offline          # fast functional + invariants + edge + conversations
./qa_suite/run_qa.sh slow             # big batteries (3000 fuzz, 2000 real GoEmotions, ...)
./qa_suite/run_qa.sh full -vvv        # everything, with a full per-action report

# Repo unit/e2e tests (e2e smoke needs the stack)
python -m pytest                      # uses pytest.ini testpaths
```

---

## QA Test Suite (`qa_suite/`)

Self-contained pytest suite that exercises the emotion engine in-process (it imports the real
`meta_learner` helpers, no service edits). Markers/run modes via `qa_suite/run_qa.sh` (and `.bat`):
`offline` (default, no stack), `slow` (big batteries, opt-in `@slow`), `live` (`@e2e`, needs stack),
`all`, `full`, `calibrate`.

- **Functional** (classification correctness): `test_functional.py` (7 equivalence classes — clear /
  mixed / vague / minimal / modifiers / emoji + determinism / boundary), `test_messages.py`
  (`data/edge_messages.json` authored edge cases), `test_coverage.py` (per-emotion corpus, all 28),
  `test_goemotions.py` (2000 **real** GoEmotions messages vs human gold), `test_conversation.py`
  (curated + 3000 generated dialogues: per-turn accuracy + valence-arc trajectory).
- **Non-functional**: `test_robustness.py` (3000 fuzz inputs → never crash, valid label),
  `test_invariants.py` (feature-vector shape, GoE gate cap ≤0.50, gate-vector shape, score
  normalisation), `test_live.py` (sequential context, sarcasm, payload contract, latency/throughput).
- **Scoring**: per-case asserts where unambiguous; large noisy corpora use an **aggregate accuracy
  gate** (prints misses, stays green). Bands live in `qa_suite/thresholds.py`; recalibrate via
  `calibrate.py` / `build_goemotions_corpus.py`.
- **Dual-mode**: passes with the trained model present (invariants run) *and* in rule-based fallback
  (those 3 invariants self-skip). `run_qa.sh` copies `meta_weights.pkl` out of the running container
  when the host lacks it.
- **`-vvv`**: prints a full per-action block (input, predicted+conf, expected ✓/✗, VADER, VAD, top-5,
  sarcasm, gate weights) via the terminal writer — works **without** `-s`. Reports every action, so
  target a specific file for readability.

## Dev Tooling & Offline Training

- **`agents/`** — InnerLink multi-agent CLI (`python agents/run.py "<task>"`): a `head_agent` plus
  domain `sub_agents/` with `knowledge/` + `inbox/` markdown context. Dev orchestration, not a runtime
  service.
- **`conversation_state_learner/`** — offline pipeline for the trajectory/context models: `collect.py`
  (generate conversations via Claude API + run them through the live stack), `train.py`,
  `train_sarcasm_classifier.py`, `migrate_and_train.py`; `data/runner.py:PipelineRunner` drives the WS.
- **All tests live under `qa_suite/`** — `qa_suite/unit/` holds the service unit tests
  (`test_feature_parity.py`, `test_hard_cases.py`, `test_context_pipeline.py`,
  `test_websocket_manager.py`, `test_ack_logic.py`); `qa_suite/test_e2e_smoke.py` is the
  `@e2e` pipeline smoke; `qa_suite/tools/eval_sentences.py` is the offline sentence battery.
  `pytest.ini` testpaths = `qa_suite`, so `python -m pytest` runs everything.

## Key Invariants & Gotchas

- **Feature vector parity**: `meta_learner.py:build_feature_vector` and `trainer/cycle.py` must produce the same 116-dim vector. Mismatch → `X has N features but StandardScaler expecting M` error.
- **Dimension constants**: `shared/constants.py` is the single source of truth. Key values: `ML_DIM=42`, `CDM_CTX_DIM=40`, `PRIOR_DIM=28`, `CONTEXT_DIM=74`, `FEATURE_DIM=116`, `N_CDM_STATES=15`. Use `CTX_*` named index constants — never hardcode integers. `context_engine_service` uses `CDM_CTX_DIM`; `GatingEnsembleNet` uses `CONTEXT_DIM` (the full 74-dim block).
- **Trajectory prior Redis key**: `trajectory:{conv_id}:prior` — 28-dim JSON list, feature slot [82:110]. Written by `trajectory/inference.py`, read by `aggregate_and_publish` in `central_responder/main.py`. Missing key → zeros (safe).
- **GoEmotions gate cap**: `GatingEnsembleNet.forward()` hard-caps GoEmotions gate weight at ≤0.50 via `.clamp(max=0.50)`. Enforced regardless of training data distribution.
- **Gate weights format**: `gate_weights_alpha` in WebSocket payload is [vader, bert, goe, vad, ctx] (5 elements). UI displays only first 3.
- **CSS import**: `main.jsx` imports `styles/tailwind.css`, `styles/design-system.css`, `styles/emotionBubbles.css`, and `index-v2.css` (in that order). `App.jsx` additionally imports `glass/CrystalGlass-v2.css`.
- **Model files on host**: `./.cache/` is bind-mounted to `/app/models` in `central_responder_service` and `trainer_service`. Models the trainer writes there (`meta_weights.pkl`, `samlowe_finetuned/`, `cdm_hmm.pkl`, dataset caches) are visible to the running services without a rebuild. (The `qa_suite` offline tests instead read `central_responder_service/models/meta_weights.pkl` — `run_qa.sh` copies it out of the container on demand.)
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
