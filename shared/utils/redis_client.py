import os
import redis.asyncio as redis
import json

class RedisClient:
    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.redis = None

    async def connect(self):
        self.redis = redis.Redis(host=self.host, port=self.port, decode_responses=True)
    
    async def publish_event(self, stream_key: str, event_data: dict):
        if not self.redis:
            await self.connect()
        
        # Prepare data for Redis: Simple flat dict where values are strings/numbers
        # We need to serialize nested dicts (like metadata) to JSON strings
        prepared_data = {}
        for k, v in event_data.items():
            if isinstance(v, (dict, list)):
                prepared_data[k] = json.dumps(v)
            else:
                prepared_data[k] = v
                
        await self.redis.xadd(stream_key, prepared_data)

    async def close(self):
        if self.redis:
            await self.redis.close()
