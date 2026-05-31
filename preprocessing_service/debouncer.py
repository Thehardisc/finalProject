"""
preprocessing_service/debouncer.py — Sliding-window message debouncer.

Groups rapid-fire messages from the same (user_id, conversation_id) pair into
logical bursts before they enter the NLP pipeline.

Problem solved:
  When a user splits a thought across multiple messages within a short window
  ("Hello" [Enter] "how are" [Enter] "you?"), each fragment reaches the Context
  Engine as a separate turn.  The CDM DFSM fires erratic state transitions,
  velocity spikes to ±1.0, and state residency inflates 3× for what is
  semantically one turn.

Mechanism:
  - First message in a burst   → is_continuation=False  (normal full turn)
  - Subsequent messages         → is_continuation=True   (fragment; CDM stays put)

The window is sliding (reset-on-input).  Flush is also forced immediately when
the buffer reaches DEBOUNCE_MAX_BURST messages or the total buffer age exceeds
DEBOUNCE_MAX_AGE_MS, capping worst-case additional latency.

Redis PEL safety: all buffered records remain unACK'd in the stream PEL.
If the service crashes, xautoclaim reclaims them on restart (all treated as
new turns — burst context is lost but correctness is preserved).

Configuration:
  DEBOUNCE_WINDOW_MS   — silence window before flush (default 2500 ms)
  DEBOUNCE_MAX_BURST   — forced flush at this burst size (default 5)
  DEBOUNCE_MAX_AGE_MS  — absolute max buffer lifetime (default 10000 ms)
"""

import asyncio
import time
import os
from typing import Callable, List, Tuple, Awaitable

DEBOUNCE_WINDOW_MS  = float(os.environ.get("DEBOUNCE_WINDOW_MS",  "2500"))
DEBOUNCE_MAX_BURST  = int(os.environ.get("DEBOUNCE_MAX_BURST",    "5"))
DEBOUNCE_MAX_AGE_MS = float(os.environ.get("DEBOUNCE_MAX_AGE_MS", "10000"))

# Flush callback type alias for documentation
#   items : [(record_id, message_data), ...]
#   flags : [is_continuation, ...]   — flags[0] is always False


class MessageDebouncer:
    """
    One buffer per (user_id, conversation_id) pair.

    Each buffer holds (record_id, message_data) tuples so the caller can
    ACK the Redis stream records after the flush callback completes.
    """

    def __init__(
        self,
        window_ms:  float = DEBOUNCE_WINDOW_MS,
        max_burst:  int   = DEBOUNCE_MAX_BURST,
        max_age_ms: float = DEBOUNCE_MAX_AGE_MS,
    ):
        self._window_ms  = window_ms
        self._max_burst  = max_burst
        self._max_age_ms = max_age_ms
        # key → {"items": [(record_id, data)], "task": Task|None, "first_ts": float}
        self._buffers: dict = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    async def add(
        self,
        record_id: str,
        data:      dict,
        on_flush:  Callable[
            [List[Tuple[str, dict]], List[bool]],
            Awaitable[None],
        ],
    ) -> None:
        """
        Add a stream record to the appropriate buffer.

        Starts (or slides) the flush timer.  Triggers an immediate synchronous
        flush when the buffer reaches max_burst or max_age_ms.
        """
        key = self._buf_key(data)
        now = time.monotonic()

        if key not in self._buffers:
            self._buffers[key] = {"items": [], "task": None, "first_ts": now}

        buf = self._buffers[key]
        buf["items"].append((record_id, data))

        age_ms = (now - buf["first_ts"]) * 1000.0
        if len(buf["items"]) >= self._max_burst or age_ms >= self._max_age_ms:
            # Hard limit reached — flush immediately, don't bother with a timer
            if buf["task"] and not buf["task"].done():
                buf["task"].cancel()
            await self._flush(key, on_flush)
            return

        # Sliding window: cancel old timer, arm a new one
        if buf["task"] and not buf["task"].done():
            buf["task"].cancel()
        buf["task"] = asyncio.create_task(self._flush_after(key, on_flush))

    async def flush_all(
        self,
        on_flush: Callable[[List[Tuple[str, dict]], List[bool]], Awaitable[None]],
    ) -> None:
        """Drain every pending buffer.  Call before graceful shutdown."""
        for key in list(self._buffers.keys()):
            await self._flush(key, on_flush)

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _buf_key(data: dict) -> str:
        return f"{data.get('user_id', '_')}:{data.get('conversation_id', '_')}"

    async def _flush_after(self, key: str, on_flush: Callable) -> None:
        try:
            await asyncio.sleep(self._window_ms / 1000.0)
        except asyncio.CancelledError:
            # A newer message arrived and reset the timer; do nothing here.
            return
        await self._flush(key, on_flush)

    async def _flush(self, key: str, on_flush: Callable) -> None:
        buf = self._buffers.pop(key, None)
        if not buf or not buf["items"]:
            return
        items = buf["items"]
        # flags[0] is always False (the burst anchor is a full turn)
        flags = [False] + [True] * (len(items) - 1)
        await on_flush(items, flags)
