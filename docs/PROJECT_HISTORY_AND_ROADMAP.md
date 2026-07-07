# Project History & Roadmap
## finalProject — Emotion AI Dashboard

---

## What Was Built This Session

### 1. Runner Fix (conversation_state_learner/data/runner.py)
- **Problem**: All messages were sent from Alice only — Bob never appeared as sender
- **Fix**: Alternating sender per turn index — even turns = Alice, odd turns = Bob

### 2. ConversationLSTM Retrain
- Collected 180 synthetic conversations (12 trajectory types × 15 each)
- New model: hidden=128, layers=2, epoch=83, val_loss=0.0069, Top-1=47.8%
- Saved to: `central_responder_service/models/trajectory_lstm.pt`

### 3. Design System v2 Integration
- Replaced `index.css` → `index-v2.css`
- Replaced `CrystalGlass.css` → `CrystalGlass-v2.css`
- Added Inter font (Google Fonts) to `index.html`

### 4. Live Demo Runner (frontend_service/src/components/DemoRunner.jsx)
- 6 hardcoded scenarios: escalating_conflict, emotional_support, growing_excitement, de_escalation, gradual_sadness, volatile_mix
- Authenticates as Alice + Bob, creates a real conversation, sends messages with 3200ms delay
- Alternates sender_id per message so each person's bubble appears correctly

### 5. Live Analytics Dashboard (frontend_service/src/pages/LiveAnalyticsDashboardPage.jsx)
- Fetches real data from `GET /admin/recent-analyses?limit=500`
- KPI cards: messages analyzed, avg confidence, top emotion, unique conversations
- Line chart: confidence over last 40 messages
- Doughnut chart: top-8 emotion distribution
- Clickable emotion filter + recent messages feed
- Auto-refreshes every 30 seconds

### 6. Context Engine & LSTM Trajectory in AnalysisDrawer
- **Context Engine section**: prev_emotion → current emotion, avg_valence, topic_resonance, volatility, episodic memory
- **Trajectory section**: Top predicted next emotion + Top-5 probability bars
- Data flows from `pipeline_log` through `conversation_update_stream` → WebSocket

### 7. api_service WebSocket Payload Extended (api_service/main.py)
- Added `context_snapshot` and `lstm_trajectory` fields to `_handle_conversation_update`
- Broadcasts to ALL conversation participants (not just sender)

### 8. Git Setup
- Initialized local repo, connected to `https://github.com/Thehardisc/finalProject`
- Branch: `fix/comprehensive-audit` (never touched main)
- Merged remote-main with `--allow-unrelated-histories`, resolved 33 conflicts keeping HEAD
- Pushed successfully to GitHub

---

## Known Issues & What Can Be Fixed Next

### Priority 1 — ML Accuracy

#### A. GoEmotions Misclassifies Rhetorical Questions as "Curiosity"
- **Root cause**: BERT fine-tuned on GoEmotions dataset treats "?" as curiosity signal
- **Fix options**:
  1. Add a post-processing rule: if sentence ends with `?` AND VADER sentiment is negative → override to `annoyance` or `frustration`
  2. Fine-tune BERT on adversarial examples with rhetorical questions
  3. Use LLM reasoning service to override BERT when context contradicts label

#### B. LSTM Top-1 Accuracy is 47.8% — Needs More Data
- Only 180 conversations used for training
- **Fix**: Generate 500+ conversations per trajectory type using the local generator
- Re-run: `python conversation_state_learner/data/runner.py --collect`
- Re-train: `python conversation_state_learner/train.py`

#### C. Meta-Learner Weights Are Stale
- `meta_weights.pkl` was trained on old model outputs
- **Fix**: After LSTM retrain, retrain meta-learner:
  `python central_responder_service/training/train_meta_learner.py`

---

### Priority 2 — Pipeline Reliability

#### D. Emotion Stream Has No Dead-Letter Queue
- If `aggregation_service` crashes mid-message, the pipeline_log is lost silently
- **Fix**: Add Redis Stream consumer groups with `XACK` + retry logic in `aggregation_service/main.py`

#### E. WebSocket Reconnection Not Handled in Frontend
- If the server restarts, the WebSocket dies and the user sees nothing
- **Fix**: Add exponential backoff reconnect in `frontend_service/src/api/client.js`
  ```js
  let retryDelay = 1000;
  ws.onclose = () => setTimeout(connect, retryDelay *= 2);
  ```

#### F. Context Engine Has No Persistence
- `context_engine_service` holds state in memory — restart = all context lost
- **Fix**: Persist `ConversationContext` objects to Redis with TTL of 24h

---

### Priority 3 — UI / UX

#### G. DemoRunner Has No Progress Indicator
- User clicks "Start Demo" and nothing visible happens for 3 seconds between messages
- **Fix**: Add a progress bar showing `message X of Y` + current scenario name

#### H. Live Analytics Does Not Show Trajectory Trends
- Dashboard shows emotion distribution but not how emotions flow over time per conversation
- **Fix**: Add a Sankey diagram showing emotion transitions (prev→current) using the `trajectory` data

#### I. AnalysisDrawer Has No Error State
- If `context_snapshot` is malformed, the drawer silently shows nothing
- **Fix**: Add fallback UI with "Context data unavailable" message

#### J. No Mobile Layout
- The entire UI is desktop-only
- **Fix**: Add responsive breakpoints to `index-v2.css` for screens < 768px

---

### Priority 4 — Infrastructure

#### K. No .gitignore for node_modules
- `frontend_service/node_modules` was accidentally staged in the merge
- **Fix**: Add `node_modules/` and `dist/` to `.gitignore`

#### L. .env File Is Committed to Git
- `.env` appears in the staged files — this is a security risk
- **Fix**: Add `.env` to `.gitignore` immediately, rotate any secrets in that file

#### M. No Health Checks in docker-compose
- Services can start but be unhealthy (Redis not ready, port not bound)
- **Fix**: Add `healthcheck:` blocks to `docker-compose.yml` for each service

#### N. BERT Service Has No Batching
- Each message calls BERT individually — slow under load
- **Fix**: Add a queue in `bert_service` that batches up to 8 messages per inference call

---

## Architecture Reference

```
User Message (WebSocket)
    ↓
api_service → ingestion_service → Redis: emotion_stream
    ↓
bert_service (GoEmotions, 28 classes)
    ↓
vader_service (sentiment: pos/neg/neu/compound)
    ↓
context_engine_service (tracks conversation state, episodic memory)
    ↓
central_responder_service (ConversationLSTM → next emotion prediction)
    ↓
aggregation_service → Redis: conversation_update_stream
    ↓
api_service._handle_conversation_update → WebSocket broadcast to all participants
    ↓
Frontend AnalysisDrawer (shows emotion, context_snapshot, lstm_trajectory)
```

## Key Files

| File | Purpose |
|------|---------|
| `api_service/main.py` | WebSocket handler, conversation_update_stream consumer |
| `central_responder_service/main.py` | ConversationLSTM inference + meta-learner |
| `context_engine_service/main.py` | Conversation context tracking |
| `aggregation_service/main.py` | Merges BERT+VADER+LSTM outputs |
| `conversation_state_learner/data/runner.py` | Collects training conversations |
| `conversation_state_learner/train.py` | Trains ConversationLSTM |
| `frontend_service/src/components/DemoRunner.jsx` | Live Demo Runner UI |
| `frontend_service/src/components/AnalysisDrawer.jsx` | ML analysis panel |
| `frontend_service/src/pages/LiveAnalyticsDashboardPage.jsx` | Analytics dashboard |

## Git

- **Branch**: `fix/comprehensive-audit`
- **Remote**: `https://github.com/Thehardisc/finalProject`
- **Rule**: NEVER touch `main` — all work goes on `fix/comprehensive-audit`
