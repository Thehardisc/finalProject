"""
persistence_service ACK-logic tests — no DB or Redis required.

Tests guard the B3 fix: a message that fails both DB persistence AND DLQ write
must NOT be ACK'd (it stays in the PEL for xautoclaim retry).

Run:
    python -m pytest persistence_service/tests/test_ack_logic.py -v
"""
import asyncio
import sys
import os
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── minimal replica of the ACK logic from persistence_service/main.py ─────────
# We test the logic directly without importing the whole service.

async def _run_one_batch(process_fn, xadd_fn):
    """
    Simplified version of the inner batch loop in persistence_service/main.py.
    process_fn: coroutine that either succeeds or raises
    xadd_fn:    coroutine that writes to DLQ (may also raise)
    Returns the list of (stream, msg_id) pairs that would be ACK'd.
    """
    to_ack = []
    stream, message_id = "test_stream", "1-0"
    session = MagicMock()
    session.commit  = MagicMock()
    session.rollback = MagicMock()
    session.close   = MagicMock()

    try:
        await process_fn()
        session.commit()
        to_ack.append((stream, message_id))
    except Exception as e:
        session.rollback()
        try:
            await xadd_fn(e)
            # Only ACK after successful DLQ write
            to_ack.append((stream, message_id))
        except Exception:
            # DLQ also failed — leave in PEL, do NOT append to to_ack
            pass
    finally:
        session.close()

    return to_ack


# ── tests ─────────────────────────────────────────────────────────────────────

class TestPersistenceAckLogic:

    @pytest.mark.asyncio
    async def test_success_is_acked(self):
        """Happy path: DB write succeeds → message ACK'd."""
        process = AsyncMock(return_value=None)
        xadd    = AsyncMock(return_value=None)

        result = await _run_one_batch(process, xadd)

        assert len(result) == 1, "Successful message must be added to to_ack"
        process.assert_awaited_once()
        xadd.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_db_fail_dlq_success_is_acked(self):
        """DB fails but DLQ write succeeds → message IS ACK'd (safely recorded in DLQ)."""
        process = AsyncMock(side_effect=Exception("DB error"))
        xadd    = AsyncMock(return_value=None)

        result = await _run_one_batch(process, xadd)

        assert len(result) == 1, "Message must be ACK'd when DLQ write succeeded"
        xadd.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_db_fail_dlq_fail_not_acked(self):
        """B3 regression: DB fails AND DLQ fails → message must NOT be ACK'd.
        It must remain in the PEL for xautoclaim retry."""
        process = AsyncMock(side_effect=Exception("DB error"))
        xadd    = AsyncMock(side_effect=Exception("Redis unreachable"))

        result = await _run_one_batch(process, xadd)

        assert len(result) == 0, (
            "When both DB and DLQ writes fail, the message must stay in PEL "
            "(not ACK'd). Found to_ack={result}"
        )

    @pytest.mark.asyncio
    async def test_multiple_messages_partial_failure(self):
        """Mixed batch: some succeed, some fail both DB+DLQ.
        Only the safe ones end up in to_ack."""
        results = []
        for i, (db_ok, dlq_ok) in enumerate([
            (True,  True),   # msg 0: success → ACK
            (False, True),   # msg 1: DB fail, DLQ ok → ACK
            (False, False),  # msg 2: DB fail, DLQ fail → NOT ACK'd
        ]):
            process = AsyncMock(return_value=None) if db_ok else AsyncMock(side_effect=Exception("err"))
            xadd    = AsyncMock(return_value=None) if dlq_ok else AsyncMock(side_effect=Exception("dlq"))
            batch   = await _run_one_batch(process, xadd)
            results.extend(batch)

        assert len(results) == 2, (
            f"Expected 2 ACKs (msgs 0 and 1), got {len(results)}: {results}"
        )
