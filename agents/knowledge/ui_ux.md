# ui_ux Knowledge Base
Last updated: 2026-06-22

## Domain Ownership
**Visual design + UX for the InnerLink frontend.**
This agent is explicitly NOT constrained by the current design.

### Owned files
| File | Role |
|------|------|
| `frontend_service/src/pages/IGDashboard.jsx` | Main chat layout |
| `frontend_service/src/pages/LandingPage.jsx` | Landing/onboarding |
| `frontend_service/src/pages/AnalyticsPage.jsx` | Analytics view |
| `frontend_service/src/components/EmotionIntelligencePanel.jsx` | Emotion scores panel |
| `frontend_service/src/components/MessageItem.jsx` | Chat bubble + emotion tag |
| `frontend_service/src/components/EmotionArcChart.jsx` | Arc chart visualization |
| `frontend_service/src/components/EmotionRadarChart.jsx` | Radar chart |
| `frontend_service/src/components/CDMStateGraph.jsx` | Conversation state graph |
| `frontend_service/src/components/AnalysisDrawer.jsx` | Slide-in analysis panel |
| `frontend_service/src/glass/CrystalGlass-v2.css` | Design system — `.crystal-shell` |
| `frontend_service/src/styles/design-system.css` | Component tokens + classes |
| `frontend_service/src/styles/tokens.css` | CSS variables (colors, spacing, radius) |
| `frontend_service/src/index-v2.css` | Base styles entry (NOT index.css) |

### Key technical constraints
- **Dark mode only** — no light mode toggle, hardcoded dark design system
- **CSS entry**: `main.jsx` imports `index-v2.css` — edits to `index.css` have zero effect
- **Rebuild**: every CSS/JSX change requires `docker compose up --build frontend_service -d`
- **Scroll**: only `messagesContainerRef` scrolls — `el.scrollTop = el.scrollHeight`, NEVER `scrollIntoView`
- **Flex layout**: chat area + messages div both need `minHeight: 0`

### Screenshot tool
`screenshot.js` (puppeteer) at project root — captures live UI to `frontend_screenshot.png`.
```bash
node screenshot.js
```

## Current Design State (2026-06-22)

### What's working
- Dark-only crystal glass aesthetic (`CrystalGlass-v2.css`)
- Emotion Intelligence Panel with gate weights visualization
- EmotionArcChart for conversation trajectory
- CDMStateGraph for conversation dynamics

### Known UX gaps
| ID | Status | Description |
|----|--------|-------------|
| ISS-UX001 | OPEN | gate_weights_alpha has 5 elements [vader,bert,goe,vad,ctx] — UI shows only first 3. ctx and vad are invisible. |
| ISS-UX002 | OPEN | Conversation phase from trajectory (opening/escalation/peak/turning_point/resolution/sustained) not shown in UI anywhere. |
| ISS-UX003 | OPEN | CDM current state (which of 15 conversation states) not shown in UI. |
| ISS-UX004 | OPEN | No loading/skeleton states — UI flashes empty on WebSocket reconnect. |
| ISS-UX005 | OPEN | Mobile layout untested — IGDashboard uses fixed columns, likely broken on small screens. |

## Improvement Backlog
| Priority | Item | Notes |
|----------|------|-------|
| High | Show all 5 gate weights (add vad + ctx bars) | Requires change in EmotionIntelligencePanel.jsx |
| High | Conversation phase badge (escalation / resolution / peak) | Requires trajectory agent to expose phase in WS payload |
| Med | Skeleton loading states for panels | Prevents flash of empty content |
| Med | CDM state indicator — current of 15 states | Small chip/badge near message timestamp |
| Med | Mobile responsive layout | IGDashboard needs column collapse at <768px |
| Low | Dark/dim mode toggle (ultra-dark for night use) | Pure cosmetic, low priority |
| Low | Emotion color animations — pulse on high-intensity emotions | Would use CSS @keyframes, no JS needed |

## Design Principles (agent-specific)
- This agent is NOT bound to the current crystal-glass aesthetic
- Proposals should be evaluated on user clarity, emotional resonance, and information density — not on "matching existing style"
- Reference: Linear.app (information density), Vercel (minimalism), Spotify (dark UI + data viz)
- Every redesign proposal must include: mobile treatment, accessibility notes, estimated dev effort

## Cross-Agent Dependencies
- **api_frontend**: provides WebSocket payload schema — UI can only show what the WS sends
- **trajectory**: needs to expose `conversation_phase` in WS payload for the phase badge feature
- **meta_learner**: all 5 gate weights are available in `gate_weights_alpha` — just not shown in UI yet

## Inter-Agent Requests (Pending)
- → trajectory: expose `conversation_phase` in WebSocket payload for UI badge
- → api_frontend: confirm gate_weights_alpha[3] (vad) and [4] (ctx) are in current WS broadcast

## Recent History
- 2026-06-22: Agent created. Existing design: dark CrystalGlass-v2 aesthetic, deployed via Nginx.
- 2026-06-20: Premium dark redesign deployed (CrystalGlass-v2, EmotionArcChart, new tokens)
