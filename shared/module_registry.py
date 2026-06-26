import json
import time
import datetime
from typing import Optional

REGISTRY_KEY = "module_registry"

DEFAULT_REQUIRED: frozenset = frozenset({"go_emotions", "basic_bert", "vader"})
DEFAULT_OPTIONAL: frozenset = frozenset({"context_engine"})


class ModuleRegistry:
    def __init__(self, redis, refresh_interval: float = 30.0):
        self._redis            = redis
        self._refresh_interval = refresh_interval
        self._last_refresh     = 0.0
        self._required: frozenset = DEFAULT_REQUIRED
        self._optional: frozenset = DEFAULT_OPTIONAL
        self._all: dict           = {}

    async def refresh(self) -> dict:
        raw = await self._redis.hgetall(REGISTRY_KEY)
        if not raw:
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

        self._required = frozenset(required) if required else DEFAULT_REQUIRED
        self._optional = frozenset(optional)
        self._all      = parsed
        self._last_refresh = time.time()
        return parsed

    async def maybe_refresh(self) -> None:
        if time.time() - self._last_refresh > self._refresh_interval:
            await self.refresh()

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


async def register_module(
    redis,
    model_name: str,
    required: bool,
    schema: str,
    logger=None,
) -> None:
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
