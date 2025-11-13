from datetime import datetime
import asyncio


class RateLimiter:
    """Token bucket rate limiter for smooth request distribution"""

    def __init__(self, rate: float):
        self.rate = rate  # requests per second
        self.tokens = rate
        self.last_update = datetime.now()
        self.lock = asyncio.Lock()  # one rquest at a time

    async def acquire(self):
        async with self.lock:
            now = datetime.now()
            elapsed = (now - self.last_update).total_seconds()
            self.tokens = min(
                self.rate, self.tokens + elapsed * self.rate
            )  # refill bucket
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0
