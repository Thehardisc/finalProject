import os
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
import json

class RedisClient:
    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.redis = None

    async def connect(self):
        pool = ConnectionPool(
            host=self.host,
            port=self.port,
            decode_responses=True,
            max_connections=20,
        )
        self.redis = redis.Redis(connection_pool=pool)
        # Explicit ping to fail fast if Redis is not reachable at startup
        await self.redis.ping()

    async def publish_event(self, stream_key: str, event_data: dict):
        if not self.redis:
            await self.connect()

        # Prepare data for Redis: flat dict where values are strings/numbers.
        # Nested dicts/lists must be serialized to JSON strings.
        prepared_data = {}
        for k, v in event_data.items():
            if isinstance(v, (dict, list)):
                prepared_data[k] = json.dumps(v)
            else:
                prepared_data[k] = v

        await self.redis.xadd(stream_key, prepared_data)

    async def close(self):
        if self.redis:
            await self.redis.aclose()
