# api_frontend Knowledge Base
Last updated: 2026-06-21

## Architecture
### api_service (port 8001)
- JWT auth: `api_service/auth_utils.py` — JWT_SECRET, JWT_EXPIRY_HOURS env vars
- Admin role: user matching ADMIN_USERNAME env var
- Routes: auth.py, users.py, conversations.py, messages.py, analytics.py, admin.py
- WebSocket: `/ws/{conversation_id}` — broadcasts emotion events to all conversation participants
- WebSocket payload includes: gate_weights_alpha [vader,bert,goe,vad,ctx] (5 floats)
- INTERNAL_API_KEY required on all endpoints (X-API-Key header)

### frontend_service (port 5173 → Nginx)
- React + Vite, served as static build by Nginx in Docker
- **Entry point**: `main.jsx` imports `index-v2.css` (NOT index.css — edits to index.css have no effect)
- **CSS**: `index-v2.css` (base) + `glass/CrystalGlass-v2.css` (design system, `.crystal-shell`)
- **App shell**: `App.jsx` wraps in `<div className="crystal-shell">`
- **Main view**: `src/pages/IGDashboard.jsx` — Instagram-style layout
- **Scroll rule**: body + `.crystal-shell` are `overflow:hidden`. Only `messagesContainerRef` scrolls.
  Use `el.scrollTop = el.scrollHeight` — NOT scrollIntoView (picks wrong ancestor).
- **Flex layout**: chat area needs `minHeight: 0`; messages div needs `minHeight: 0` for overflow to activate.
- **Dark mode**: always dark — no light mode toggle (hardcoded dark design system)

### Frontend rebuild required
CSS + JSX changes need: `docker compose up --build frontend_service -d`
(Nginx serves a pre-built Vite bundle — live reload not available in Docker)

### Session reset cleanup (conversations.py)
Deletes per-speaker CDM keys on session reset:
```python
async for key in r.scan_iter(f"conv:{conversation_id}:spk:*"):
    await r.delete(key)
```

## Known Issues
- gate_weights_alpha has 5 elements [vader,bert,goe,vad,ctx] but UI only displays first 3.
- Conversation phase from trajectory not exposed in WebSocket payload — only in server logs.

## Improvement Queue
- **[High]** Display conversation phase (trajectory) in UI — "escalation" / "resolution" badge.
- **[Med]** Show ctx gate weight in UI alongside vader/bert/goe weights (5th bar).
- **[Med]** Add CDM current state indicator to UI (which of 15 states the conversation is in).
- **[Low]** WebSocket reconnect logic — currently no exponential backoff on disconnect.

## Cross-Agent Dependencies
- Provides: user-facing interface consuming **meta_learner** emotion predictions
- Provides: REST routes that clear **context_engine** per-speaker Redis keys on session reset
- Depends on: **pipeline** for message ingestion (POST /messages → ingestion_service)
- Depends on: **infra** for PostgreSQL (message history) and Redis (WebSocket state)

## Inter-Agent Requests (Pending)
- → trajectory: expose conversation phase in WebSocket payload
- → meta_learner: expose full 5-element gate_weights_alpha in UI

## Recent History
- 2026-06-20: Dark-only premium redesign deployed (CrystalGlass-v2)
- 2026-06-20: Session reset now cleans per-speaker CDM keys (spk:* pattern)
- 2026-06-20: Scroll behavior fixed: el.scrollTop = el.scrollHeight instead of scrollIntoView
