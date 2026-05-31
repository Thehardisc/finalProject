# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the System

**Prerequisites:** Docker Desktop running, `.env` file created from `.env-template`:
```bash
cp .env.example .env
# Edit .env: set INTERNAL_KEY, JWT_SECRET, ADMIN_USERNAME
```

**Start (auto-detects GPU):**
```bash
bash start.sh          # macOS/Linux
start.bat              # Windows
```

**Start with explicit GPU:**
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

**Start CPU-only:**
```bash
docker compose up --build
```

**Rebuild a single service:**
```bash
docker compose up --build <service_name>
# e.g. docker compose up --build central_service
```

**View logs:**
```bash
docker compose logs -f <service_name>
docker compose logs -f central_service
```

## Service Endpoints

| Service | URL |
|---|---|
| Frontend (React) | http://localhost:5173 |
| API (REST + WebSocket) | http://localhost:8001 |
| Ingestion | http://localhost:8000 |

## Architecture

**Eleven services** connected by **Redis Streams** (not direct HTTP calls):

```
[User] → Frontend(5173) → API(8001) → ingestion(8000)
                                           ↓
                                     [redis: message_stream]
                                           ↓
                                    preprocessing_service
                                           ↓
                              [redis: preprocessed_stream]
                           ↙          ↓          ↘
                    vader_service  bert_service  gpt_service
                                  └─────────────────────────────────
                                                   ↓
                                [redis: partial_analysis_stream]
                                                   ↓
                                      central_service_service
                                     (+ context_service_service)
                                                   ↓
                                     [redis: emotion_stream]
                                                   ↓
                              ┌───────────────────────────┐
                       aggregation_service        llm_reasoning_service
                              ↓
                [redis: conversation_update_stream]
                         ↙              ↘
             persist_service        api_service → WebSocket → Frontend
```

## Critical Architecture Rules

**The central_service_service** is the most complex service. It:
1. Waits for **three** services to report to `partial_analysis_stream`: `vader`, `bert`, `gpt`
2. Builds a **87-dim feature vector**: `[0:4]` VADER + `[4:11]` BERT + `[11:39]` GOT + `[39:87]` Context Engine
3. Runs a **sklearn LogisticRegression pipeline** (the "meta-learner") on this vector
4. Also runs an optional **LSTM trajectory model** for next-emotion prediction
5. Contains a **background thread** that retrains the meta-learner on new data every 1800s

**context_engine_service** is separate and also reads from `preprocessed_stream`. It publishes a 48-dim vector to `partial_analysis_stream` with `model_name=context_engine`. The central_service waits for the 3 required models, then optionally uses the context vector if it arrived.

**The 48-dim context vector layout** (from `context_engine_service/main.py`):
- `[0]` = historical_valence (from Vdb vector search)
- `[1]` = topic_resonance (from Vdb)
- `[2]` = volatility (exponential moving average)
- `[3]` = current_valence (from redis)
- `[4]` = message length
- `[5]` = latency_ms
- `[6:48]` = SentenceTransformer embedding[:42] (model: `all-MiniLM-L6-2`)

**The meta-learner model** lives at `central_service_service/models/meta_weights.pkl`. If it doesn't exist or doesn't match the current feature dimensions (87), the service falls back to rule-based logic. Deleting this file forces retraining on the next startup.

**[Shared constants](central_service_service/shared/constants.py)** is the single source of truth for feature dimensions:
- `FEATURE_DIM = 87` (must match the model and all feature-building code)
- `FEATURE_DIM = 48` (context engine)
- `FEATURE_DIM = 39` (meta-learner)

If you change any of these, you must also delete `meta_weights.pkl` and `dataset_features_cache.pkl` to force a full retrain.

## Key Files

| File | Purpose |
|---|---|
| `central_service_service/meta_learner.py` | Feature vector construction + sklearn inference |
| `central_service_service/main.py` | Main stream processing loop + aggregation |
| `central_service_service/trainer.py` | Background retraining logic |
| `central_service_service/trajectory/inference.py` | LSTM trajectory model |
| `context_engine_service/main.py` | 48-dim context vector builder |
| `shared/constants.py` | Feature dimensions and label lists |
| `api_service/main.py` | REST endpoints + WebSocket server |
| `api_service/websocket/listener.py` | Redis → WebSocket bridge |

## Data Storage

- **Redis** (`redis:6379`): All inter-service messaging (streams), session state, rate limiting
- **PostgreSQL** (`db:5432`): Persisted messages, emotion results, user accounts
- **Qddb** (`qddb:6333`): Long-term semantic memory (used by context_engine_service)

Redis streams used (in order):
1. `message_stream`
2. `preprocessed_stream`
3. `partial_analysis_stream`
4. `emotion_stream`
5. `conversation_update_stream`
6. `conversation_update_stream`

## Auth

The system uses two auth mechanisms:
- **API key** (`X-API-Key` header): Required for the ingestion endpoint. Set via `INTERNAL_API_KEY` env var.
- **JWT**: Used for user-facing routes on the API. Set via `JWT_SECRET` env var.

## Frontend

- **React + Vite** in `frontend_service/`
- WebSocket connection in `src/hooks/useWebSocket.js`
- The "status wall" system waits for `systems_ready` before allowing chat
- Zero dark mode: all color comes from emotion values, never from a dark background
- Emotional colors are defined in `src/components/EmotionPalette.js`

## Model Files

Model files in `central_service_service/models/`:
- `meta_weights.pkl` — the trained sklearn pipeline (87-dim input)
- `meta_weights_meta.json` — training metadata
- `dataset_features_cache.pkl` — cached features from GOT (with `feature_dim` key for validation)
- `.ready` — sentinel file; presence means a valid model is loaded
- `trajectory_lstm.pt` — optional LSTM model
- `trajectory_config.json` — LSTM architecture config

None of these are committed to git. They are generated at runtime.
