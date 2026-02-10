from __future__ import annotations

import os
import time
from typing import Optional, Callable

import redis
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str, limit_per_min: int = 120):
        super().__init__(app)
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        self.limit = limit_per_min

    async def dispatch(self, request: Request, call_next):
        # Skip for docs/static
        if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi.json") or request.url.path.startswith("/redoc"):
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        key = f"rl:{ip}:{int(time.time() // 60)}"
        try:
            n = self.r.incr(key)
            if n == 1:
                self.r.expire(key, 70)
            if n > self.limit:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too Many Requests", "limit_per_min": self.limit},
                )
        except Exception:
            # If redis fails, don't block classroom demo
            pass

        return await call_next(request)
