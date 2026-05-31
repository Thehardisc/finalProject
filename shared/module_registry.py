"""
shared/module_registry.py — Dynamic Module Registry backed by a Redis Hash.

Services self-register on startup via register_module().
ModuleRegistry is consumed by central_responder_service to replace the
hardcoded expected_models list with a dynamically discovered set.

Redis layout:
  Key   : "module_registry"     (Hash)
  Fields: model_name → JSON blob
  Blob  : {"required": bool, "schema": "scores"|"context_vector",
            "registered_at": "ISO-8601"}

Required modules block aggregation until all are present.
Optional modules are given OPTIONAL_TIMEOUT_MS (default 50 ms) after all
required models arrive; if they haven't appeared by then, aggregation proceeds
without them.

Fallback: if the registry Hash is empty in Redis (race at startup), the module
uses DEFAULT_REQUIRED / DEFAULT_OPTIONAL so the system doesn't stall.
"""

import json
import time
import datetime
from typing import Optional

REGISTRY_KEY = "module_registry"

# Backward-compat fallback — mirrors the hardcoded list that existed before v2.
DEFAULT_REQUIRED: frozenset = frozenset({"go_emotions", "basic_bert", "vader"})
DEFAULT_OPTIONAL: frozenset = frozenset({"context_engine"})


class ModuleRegistry:
    """
    Lightweight, cache-first view of the Redis module_registry Hash.

    All property accesses are synchronous (reads from local cache).
    Actual Redis I/O only happens inside async refresh() / maybe_refresh().
    """

    def __init__(self, redis, refresh_interval: float = 30.0):
        self._redis            = redis
        self._refresh_interval = refresh_interval
        self._last_refresh     = 0.0
        self._required: frozenset = DEFAULT_REQUIRED
        self._optional: frozenset = DEFAULT_OPTIONAL
        self._all: dict           = {}

    # ── Async refresh ──────────────────────────────────────────────────────────

    async def refresh(self) -> dict:
        """
        Pull current registry from Redis and update local cache.
        Falls back to defaults if the hash is empty.
        """
        raw = await self._redis.hgetall(REGISTRY_KEY)
        if not raw:
            # No services registered yet (startup race) — keep defaults
            self._last_refresh = time.time()
            return {}

        required, optional, parsed = set(), set(), {}
        for model_name, blob in raw.items():
            try:
                info = json.loads(blob)
                parsed[model_name] = info
                (required if info.get("required", False) else optional).add(model_name)
            except (json.JSONDecodeError, TypeError):
                pass

        # Never leave required empty — fallback keeps the system running
        self._required = frozenset(required) if required else DEFAULT_REQUIRED
        self._optional = frozenset(optional)
        self._all      = parsed
        self._last_refresh = time.time()
        return parsed

    async def maybe_refresh(self) -> None:
        """Refresh only when the local cache is stale. O(1) on the hot path."""
        if time.time() - self._last_refresh > self._refresh_interval:
            await self.refresh()

    # ── Synchronous accessors (hot-path safe) ─────────────────────────────────

    @property
    def required(self) -> frozenset:
        return self._required

    @property
    def optional(self) -> frozenset:
        return self._optional

    def all_required_present(self, received: set) -> bool:
        return self._required.issubset(received)

    def all_optional_present(self, received: set) -> bool:
        return self._optional.issubset(received)

    def missing_required(self, received: set) -> set:
        return self._required - received

    def log_state(self, logger) -> None:
        logger.info(
            f"[Registry] required={set(self._required)}  "
            f"optional={set(self._optional)}"
        )


# ── Service-side self-registration ─────────────────────────────────────────────

async def register_module(
    redis,
    model_name: str,
    required: bool,
    schema: str,
    logger=None,
) -> None:
    """
    Called once at service startup after the Redis connection is established.

    Args:
        redis:      live async Redis connection (or any object with .hset())
        model_name: the model_name this service publishes to partial_analysis_stream
        required:   True → aggregation blocks until this model reports
                    False → aggregation waits at most OPTIONAL_TIMEOUT_MS ms
        schema:     "scores"          — dict of emotion label → probability
                    "context_vector"  — 23-dim float list + named scalar fields
        logger:     optional logger for the confirmation message
    """
    payload = json.dumps({
        "required":      required,
        "schema":        schema,
        "registered_at": datetime.datetime.utcnow().isoformat() + "Z",
    })
    await redis.hset(REGISTRY_KEY, model_name, payload)
    msg = (
        f"[Registry] '{model_name}' registered  "
        f"required={required}  schema={schema}"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)
