import os
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
import json

STREAM_MAXLEN = int(os.getenv("REDIS_STREAM_MAXLEN", 10_000))


class RedisClient:
    def __init__(self):
        self.host     = os.getenv("REDIS_HOST",     "localhost")
        self.port     = int(os.getenv("REDIS_PORT", 6379))
        self.password = os.getenv("REDIS_PASSWORD", None)
        self.redis    = None

    async def connect(self):
        pool = ConnectionPool(
            host=self.host,
            port=self.port,
            password=self.password,
            decode_responses=True,
            max_connections=20,
        )
        self.redis = redis.Redis(connection_pool=pool)
        await self.redis.ping()

    async def publish_event(self, stream_key: str, event_data: dict):
        if not self.redis:
            await self.connect()

        prepared_data = {}
        for k, v in event_data.items():
            if isinstance(v, (dict, list)):
                prepared_data[k] = json.dumps(v)
            else:
                prepared_data[k] = v

        await self.redis.xadd(stream_key, prepared_data, maxlen=STREAM_MAXLEN, approximate=True)

    async def close(self):
        if self.redis:
            await self.redis.aclose()
            self.redis = None
