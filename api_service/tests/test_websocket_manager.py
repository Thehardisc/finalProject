"""
api_service/websocket/manager.py tests.

Run:
    python -m pytest api_service/tests/test_websocket_manager.py -v
"""
import sys, os, pytest
from unittest.mock import AsyncMock, MagicMock

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Stub fastapi so manager.py imports without the full stack
import types
if "fastapi" not in sys.modules:
    sys.modules["fastapi"] = types.ModuleType("fastapi")
    sys.modules["fastapi"].WebSocket = object

from api_service.websocket.manager import ConnectionManager


# ── helpers ──────────────────────────────────────────────────────────────────

def _ws(send_ok=True):
    ws = AsyncMock()
    if send_ok:
        ws.send_json = AsyncMock(return_value=None)
    else:
        ws.send_json = AsyncMock(side_effect=Exception("connection closed"))
    ws.accept = AsyncMock(return_value=None)
    return ws


# ── tests ─────────────────────────────────────────────────────────────────────

class TestConnectionManager:

    @pytest.mark.asyncio
    async def test_connect_adds_to_active(self):
        mgr = ConnectionManager()
        ws  = _ws()
        await mgr.connect(ws, "alice")
        assert "alice" in mgr.active_connections
        assert ws in mgr.active_connections["alice"]

    @pytest.mark.asyncio
    async def test_disconnect_removes_socket(self):
        mgr = ConnectionManager()
        ws  = _ws()
        await mgr.connect(ws, "alice")
        mgr.disconnect(ws, "alice")
        assert "alice" not in mgr.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_sockets(self):
        mgr = ConnectionManager()
        ws1, ws2 = _ws(), _ws()
        await mgr.connect(ws1, "bob")
        await mgr.connect(ws2, "bob")
        await mgr.broadcast_to_user("bob", {"type": "ping"})
        ws1.send_json.assert_awaited_once_with({"type": "ping"})
        ws2.send_json.assert_awaited_once_with({"type": "ping"})

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connection(self):
        """B4 regression: a failed send must evict the dead socket."""
        mgr  = ConnectionManager()
        good = _ws(send_ok=True)
        dead = _ws(send_ok=False)
        await mgr.connect(good, "carol")
        await mgr.connect(dead, "carol")

        await mgr.broadcast_to_user("carol", {"type": "ping"})

        # Dead connection must be removed
        assert dead not in mgr.active_connections.get("carol", []), \
            "Dead socket must be evicted from active_connections after send failure"
        # Good connection must stay
        assert good in mgr.active_connections.get("carol", []), \
            "Live socket must remain in active_connections"

    @pytest.mark.asyncio
    async def test_broadcast_all_dead_clears_user_entry(self):
        """If all connections for a user die, the user key must be removed."""
        mgr  = ConnectionManager()
        dead = _ws(send_ok=False)
        await mgr.connect(dead, "dave")
        await mgr.broadcast_to_user("dave", {"type": "ping"})
        assert "dave" not in mgr.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_unknown_user_is_noop(self):
        """Broadcasting to a user with no connections must not raise."""
        mgr = ConnectionManager()
        await mgr.broadcast_to_user("nobody", {"type": "ping"})  # should not raise

    @pytest.mark.asyncio
    async def test_multiple_connects_same_user(self):
        """Multiple tabs / devices for the same user accumulate correctly."""
        mgr = ConnectionManager()
        ws1, ws2, ws3 = _ws(), _ws(), _ws()
        await mgr.connect(ws1, "eve")
        await mgr.connect(ws2, "eve")
        await mgr.connect(ws3, "eve")
        assert len(mgr.active_connections["eve"]) == 3

    @pytest.mark.asyncio
    async def test_disconnect_unknown_user_is_noop(self):
        """Disconnecting a socket that was never added must not raise."""
        mgr = ConnectionManager()
        mgr.disconnect(_ws(), "ghost")  # should not raise
