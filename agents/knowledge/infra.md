# infra Knowledge Base
Last updated: 2026-06-21

## Architecture
### Services (docker-compose.yml)
| Service                   | Port  | Notes                                      |
|--------------------------|-------|--------------------------------------------|
| ingestion_service         | 8000  | REST entry point                           |
| preprocessing_service     | —     | text normalization + debouncer             |
| vader_service             | —     | lexicon sentiment                          |
| bert_service              | —     | 7-class Ekman                              |
| goemotions_service        | —     | 28-class GoEmotions + emoji                |
| central_responder_service | 9090  | meta-learner + trainer + trajectory LSTM   |
| aggregation_service       | —     | conversation state tracking                |
| llm_reasoning_service     | —     | optional LLM layer (default: RULE_BASED)   |
| context_engine_service    | —     | CDM HMM + Qdrant episodic memory           |
| persistence_service       | —     | PostgreSQL writer                          |
| api_service               | 8001  | WebSocket + REST                           |
| frontend_service          | 5173  | React (Nginx static)                       |
| qdrant                    | 6333  | vector DB                                  |
| redis                     | 6379  | streams + state hashes                     |
| db (postgres)             | 5432  | long-term storage                          |

### Shared Constants (shared/constants.py) — SINGLE SOURCE OF TRUTH
```python
FEATURE_DIM    = 116   # total feature vector size
ML_DIM         = 42    # NLP block (VADER+BERT+GoE+VAD)
CDM_CTX_DIM    = 40    # CDM context block
PRIOR_DIM      = 28    # trajectory prior block
CONTEXT_DIM    = 74    # full context block (CDM+Prior+Sarcasm+Dynamics+Appraisal)
N_CDM_STATES   = 15    # HMM states
SARCASM_DIM    = 1
DYNAMICS_DIM   = 2
APPRAISAL_DIM  = 3
```
Use CTX_* named index constants — NEVER hardcode integers.

### Log rotation (x-log-rotation YAML anchor)
All services: json-file driver, max-size=10m, max-file=3. ~30MB per container.

### Logging env vars
- LOG_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL (default INFO)
- LOG_FORMAT: TEXT|JSON (default TEXT)

### Model files (bind-mounted, no rebuild needed)
- `central_responder_service/models/` → `/app/models/`
- `.cache/` → trainer cache directory

### Key env vars
```
TZ=Asia/Jerusalem
INTERNAL_API_KEY=...
JWT_SECRET=...
REDIS_PASSWORD=        # empty for dev
RETRAIN_INTERVAL_SECONDS=1800
ACCURACY_GATE=0.40
MAX_SAMPLES=2500
LLM_PROVIDER=RULE_BASED
ADMIN_USERNAME=admin
OPTIONAL_TIMEOUT_MS=1000
```

### aggregation_service
Writes `conv:{cid}:spk:{uid}:valence_seq` (JSON list, oldest-first) after each message.
Also updates `conversation:{cid}` hash with per-speaker valence for speaker_divergence calc.

## Known Issues
- **[Fixed 2026-06-21]** start.sh indentation bug — bash silent exit due to indent error.

## Improvement Queue
- **[Med]** Health check endpoints on all services — docker-compose healthcheck: currently missing on most.
- **[Low]** GPU docker-compose.gpu.yml — verify it works with current model architecture.
- **[Low]** Add Prometheus scrape config to docker-compose for central_responder :9090 metrics.

## Cross-Agent Dependencies
- Provides: Redis, PostgreSQL, Qdrant, docker networking to ALL agents
- Provides: shared/constants.py as single source of truth for dimension values
- Provides: aggregation_service → conv:{cid}:spk:{uid}:valence_seq for **context_engine**

## Inter-Agent Requests (Pending)
*None*

## Recent History
- 2026-06-21: start.sh indent fix applied
- 2026-06-20: Per-speaker valence_seq writing added to aggregation_service
- 2026-06-20: Session reset in conversations.py now also deletes spk:* Redis keys
