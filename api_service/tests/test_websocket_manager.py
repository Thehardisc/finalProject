import sys, os, pytest, types
from unittest.mock import AsyncMock, MagicMock

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

for _mod in ["fastapi", "fastapi.middleware", "fastapi.middleware.cors",
             "fastapi.staticfiles", "fastapi.responses",
             "pydantic", "pydantic.fields",
             "sqlalchemy", "sqlalchemy.orm",
             "shared.utils.redis_client", "shared.utils.logger",
             "shared.utils.auth", "shared.constants",
             "api_service.auth_utils", "api_service.db.pool",
             "api_service.routes.conversations", "api_service.routes.messages"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

fa = sys.modules["fastapi"]
fa.FastAPI           = MagicMock(return_value=MagicMock())
fa.WebSocket         = object
fa.WebSocketDisconnect = Exception
fa.HTTPException     = Exception
fa.Depends           = lambda x: x
fa.Query             = MagicMock()
fa.status            = MagicMock()
fa.Response          = MagicMock()

sys.modules["shared.utils.logger"].get_logger = lambda name: MagicMock()
sys.modules["shared.utils.logger"].sanitize_email = lambda x: x
sys.modules["shared.constants"].EMOTION_LABELS = []

from api_service.websocket.manager import ConnectionManager as WsConnectionManager

import ast as _ast
_main_src = open(os.path.join(_ROOT, "api_service", "main.py")).read()
_globs: dict = {
    "WebSocket": object,
    "logger": MagicMock(),
}
_tree = _ast.parse(_main_src)
for node in _tree.body:
    if isinstance(node, _ast.ClassDef) and node.name == "ConnectionManager":
        exec(compile(_ast.Module(body=[node], type_ignores=[]), "<main_cm>", "exec"), _globs)
        break
MainConnectionManager = _globs["ConnectionManager"]


def _ws(send_ok=True):
    ws = AsyncMock()
    ws.send_json = AsyncMock(return_value=None) if send_ok else AsyncMock(
        side_effect=Exception("connection closed")
    )
    ws.accept = AsyncMock(return_value=None)
    return ws


def _make_suite(manager_cls):
    class Suite:
        @pytest.mark.asyncio
        async def test_connect_adds_to_active(self):
            mgr = manager_cls()
            ws  = _ws()
            await mgr.connect(ws, "alice")
            assert "alice" in mgr.active_connections
            assert ws in mgr.active_connections["alice"]

        @pytest.mark.asyncio
        async def test_disconnect_removes_socket(self):
            mgr = manager_cls()
            ws  = _ws()
            await mgr.connect(ws, "alice")
            mgr.disconnect(ws, "alice")
            assert "alice" not in mgr.active_connections

        @pytest.mark.asyncio
        async def test_broadcast_sends_to_all_sockets(self):
            mgr = manager_cls()
            ws1, ws2 = _ws(), _ws()
            await mgr.connect(ws1, "bob")
            await mgr.connect(ws2, "bob")
            await mgr.broadcast_to_user("bob", {"type": "ping"})
            ws1.send_json.assert_awaited_once_with({"type": "ping"})
            ws2.send_json.assert_awaited_once_with({"type": "ping"})

        @pytest.mark.asyncio
        async def test_broadcast_removes_dead_connection(self):
            mgr  = manager_cls()
            good = _ws(send_ok=True)
            dead = _ws(send_ok=False)
            await mgr.connect(good, "carol")
            await mgr.connect(dead, "carol")
            await mgr.broadcast_to_user("carol", {"type": "ping"})
            assert dead not in mgr.active_connections.get("carol", []), \
                "Dead socket must be evicted"
            assert good in mgr.active_connections.get("carol", []), \
                "Live socket must remain"

        @pytest.mark.asyncio
        async def test_broadcast_all_dead_clears_user_entry(self):
            mgr  = manager_cls()
            dead = _ws(send_ok=False)
            await mgr.connect(dead, "dave")
            await mgr.broadcast_to_user("dave", {"type": "ping"})
            assert "dave" not in mgr.active_connections

        @pytest.mark.asyncio
        async def test_broadcast_unknown_user_is_noop(self):
            mgr = manager_cls()
            await mgr.broadcast_to_user("nobody", {"type": "ping"})

        @pytest.mark.asyncio
        async def test_multiple_connects_same_user(self):
            mgr = manager_cls()
            ws1, ws2, ws3 = _ws(), _ws(), _ws()
            await mgr.connect(ws1, "eve")
            await mgr.connect(ws2, "eve")
            await mgr.connect(ws3, "eve")
            assert len(mgr.active_connections["eve"]) == 3

        @pytest.mark.asyncio
        async def test_disconnect_unknown_user_is_noop(self):
            mgr = manager_cls()
            mgr.disconnect(_ws(), "ghost")

    return Suite


class TestConnectionManager(_make_suite(WsConnectionManager)):
    pass


class TestMainConnectionManager(_make_suite(MainConnectionManager)):
    pass
