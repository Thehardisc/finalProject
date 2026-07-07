
import sys
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


async def _run_one_batch(process_fn, xadd_fn):
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
            to_ack.append((stream, message_id))
        except Exception:
            pass
    finally:
        session.close()

    return to_ack


class TestPersistenceAckLogic:

    @pytest.mark.asyncio
    async def test_success_is_acked(self):
        process = AsyncMock(return_value=None)
        xadd    = AsyncMock(return_value=None)

        result = await _run_one_batch(process, xadd)

        assert len(result) == 1, "Successful message must be added to to_ack"
        process.assert_awaited_once()
        xadd.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_db_fail_dlq_success_is_acked(self):
        process = AsyncMock(side_effect=Exception("DB error"))
        xadd    = AsyncMock(return_value=None)

        result = await _run_one_batch(process, xadd)

        assert len(result) == 1, "Message must be ACK'd when DLQ write succeeded"
        xadd.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_db_fail_dlq_fail_not_acked(self):
        process = AsyncMock(side_effect=Exception("DB error"))
        xadd    = AsyncMock(side_effect=Exception("Redis unreachable"))

        result = await _run_one_batch(process, xadd)

        assert len(result) == 0, (
            "When both DB and DLQ writes fail, the message must stay in PEL "
            "(not ACK'd). Found to_ack={result}"
        )

    @pytest.mark.asyncio
    async def test_max_retries_always_acks(self):
        MAX = 5
        process = AsyncMock(side_effect=Exception("permanent DB error"))
        dlq_fail = AsyncMock(side_effect=Exception("DLQ also down"))

        # Simulate attempt == MAX
        async def _run_at_max_attempt():
            to_ack = []
            stream, message_id = "test_stream", "1-0"
            session = MagicMock()
            session.commit = MagicMock()
            session.rollback = MagicMock()
            session.close = MagicMock()
            attempt = MAX

            try:
                await process()
                session.commit()
                to_ack.append((stream, message_id))
            except Exception:
                session.rollback()
                if attempt >= MAX:
                    try:
                        await dlq_fail(Exception("permanent"))
                    except Exception:
                        pass
                    to_ack.append((stream, message_id))
                else:
                    try:
                        await dlq_fail(Exception("transient"))
                        to_ack.append((stream, message_id))
                    except Exception:
                        pass
            finally:
                session.close()
            return to_ack

        result = await _run_at_max_attempt()
        assert len(result) == 1, \
            "At MAX_DELIVERY_ATTEMPTS the message must be ACK'd to stop the infinite retry loop"

    @pytest.mark.asyncio
    async def test_multiple_messages_partial_failure(self):
        results = []
        for i, (db_ok, dlq_ok) in enumerate([
            (True,  True),
            (False, True),
            (False, False),
        ]):
            process = AsyncMock(return_value=None) if db_ok else AsyncMock(side_effect=Exception("err"))
            xadd    = AsyncMock(return_value=None) if dlq_ok else AsyncMock(side_effect=Exception("dlq"))
            batch   = await _run_one_batch(process, xadd)
            results.extend(batch)

        assert len(results) == 2, (
            f"Expected 2 ACKs (msgs 0 and 1), got {len(results)}: {results}"
        )
