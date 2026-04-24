import os
import time
from fastapi import Header, HTTPException, status
from shared.utils.logger import get_logger

logger = get_logger("shared.auth")

# api key validation
# Default key for internal use/local dev. 
# In production, set INTERNAL_API_KEY environment variable.
VALID_API_KEYS = os.environ.get("INTERNAL_API_KEY", "dev-secret-key").split(",")

async def validate_api_key(x_api_key: str = Header(None)):
    """
    FastAPI dependency to validate X-API-Key header.
    """
    if not x_api_key:
        logger.warning("Request missing X-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Missing API Key"
        )
    
    if x_api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API Key attempt: {x_api_key[:4]}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid API Key"
        )
    
    return x_api_key

# rate limiter
class RateLimiter:
    """
    Simple Redis-backed Fixed Window Rate Limiter.
    """
    def __init__(self, redis_client, limit: int = 60, window: int = 60):
        """
        :param limit: Max requests allowed in the window.
        :param window: Window size in seconds.
        """
        self.redis_client = redis_client
        self.limit = int(os.environ.get("RATE_LIMIT_MAX", limit))
        self.window = int(os.environ.get("RATE_LIMIT_WINDOW", window))

    async def is_allowed(self, identifier: str) -> bool:
        """
        Checks if the given identifier (e.g., API key or IP) is within limits.
        """
        r = self.redis_client.redis
        if not r:
            # If Redis is down, we fail open for availability but log an error
            logger.error("RateLimiter: Redis connection missing. Failing open.")
            return True
        
        # Key format: ratelimit:<identifier>:<window_bucket>
        bucket = int(time.time() // self.window)
        key = f"ratelimit:{identifier}:{bucket}"
        
        try:
            count = await r.incr(key)
            if count == 1:
                await r.expire(key, self.window + 10) # 10s buffer
            
            if count > self.limit:
                logger.warning(f"Rate limit exceeded for {identifier}: {count}/{self.limit}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"RateLimiter error: {e}")
            return True # Fail open
