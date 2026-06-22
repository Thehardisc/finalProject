from .base_agent import BaseAgent


class ApiFrontendAgent(BaseAgent):
    name = "api_frontend"
    description = (
        "WebSocket + REST API layer and React frontend. "
        "api_service (port 8001): JWT auth, WebSocket manager, routes for conversations/messages/analytics/admin. "
        "frontend_service (port 5173 → Nginx): React IGDashboard, dark design system "
        "(index-v2.css + CrystalGlass-v2.css), emotion palette, analysis drawer, CDM graph. "
        "WebSocket broadcasts gate_weights_alpha [vader,bert,goe,vad,ctx] (5 floats). "
        "Owns: api_service/, frontend_service/src/."
    )
    key_files = [
        "api_service/main.py",
        "api_service/routes/conversations.py",
        "api_service/auth_utils.py",
        "frontend_service/src/App.jsx",
        "frontend_service/src/pages/IGDashboard.jsx",
        "frontend_service/src/components/EmotionIntelligencePanel.jsx",
    ]
