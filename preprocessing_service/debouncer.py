import asyncio
import time
import os
from typing import Callable, List, Tuple, Awaitable

FAST_DISPATCH_MS    = float(os.environ.get("FAST_DISPATCH_MS",    "100"))
BURST_WINDOW_MS     = float(os.environ.get("BURST_WINDOW_MS",     "500"))
DEBOUNCE_MAX_BURST  = int(os.environ.get("DEBOUNCE_MAX_BURST",    "5"))
DEBOUNCE_MAX_AGE_MS = float(os.environ.get("DEBOUNCE_MAX_AGE_MS", "10000"))


class MessageDebouncer:
    def __init__(
        self,
        fast_dispatch_ms: float = FAST_DISPATCH_MS,
        burst_window_ms:  float = BURST_WINDOW_MS,
        max_burst:        int   = DEBOUNCE_MAX_BURST,
        max_age_ms:       float = DEBOUNCE_MAX_AGE_MS,
    ):
        self._fast_ms    = fast_dispatch_ms
        self._burst_ms   = burst_window_ms
        self._max_burst  = max_burst
        self._max_age_ms = max_age_ms
        self._buffers: dict = {}

    async def add(
        self,
        record_id: str,
        data:      dict,
        on_flush:  Callable[
            [List[Tuple[str, dict]], List[bool]],
            Awaitable[None],
        ],
    ) -> None:
        key = self._buf_key(data)
        now = time.monotonic()

        is_burst = key in self._buffers
        if not is_burst:
            self._buffers[key] = {"items": [], "task": None, "first_ts": now}

        buf = self._buffers[key]
        buf["items"].append((record_id, data))

        age_ms = (now - buf["first_ts"]) * 1000.0
        if len(buf["items"]) >= self._max_burst or age_ms >= self._max_age_ms:
            if buf["task"] and not buf["task"].done():
                buf["task"].cancel()
            await self._flush(key, on_flush)
            return

        if buf["task"] and not buf["task"].done():
            buf["task"].cancel()

        window = self._burst_ms if is_burst else self._fast_ms
        buf["task"] = asyncio.create_task(self._flush_after(key, on_flush, window))

    @staticmethod
    def _buf_key(data: dict) -> str:
        return f"{data.get('user_id', '_')}:{data.get('conversation_id', '_')}"

    async def _flush_after(self, key: str, on_flush: Callable, window_ms: float) -> None:
        try:
            await asyncio.sleep(window_ms / 1000.0)
        except asyncio.CancelledError:
            return
        await self._flush(key, on_flush)

    async def _flush(self, key: str, on_flush: Callable) -> None:
        buf = self._buffers.pop(key, None)
        if not buf or not buf["items"]:
            return
        items = buf["items"]
        flags = [False] + [True] * (len(items) - 1)
        await on_flush(items, flags)
