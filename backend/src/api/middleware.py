import json
import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)

from src.errors.base import AppError

logger = logging.getLogger(__name__)


MAX_REQUEST_BODY = 10 * 1024 * 1024  # 10 MB

# Rate limiting: per-IP, sliding window
# フロントが5秒ごとに3API + デモ操作で余裕をもたせる
_RATE_LIMIT = 300  # requests per window
_RATE_WINDOW = 60.0  # seconds
_RATE_STORE_MAX_SIZE = 10000
_rate_store: dict[str, list[float]] = {}


def _is_rate_limited(client_ip: str) -> bool:
    """Check if a client IP has exceeded the rate limit."""
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW

    timestamps = _rate_store.get(client_ip)
    if timestamps is None:
        _rate_store[client_ip] = [now]
        return False

    # Remove expired entries
    while timestamps and timestamps[0] < cutoff:
        timestamps.pop(0)

    # Evict empty keys to prevent memory leak
    if not timestamps:
        del _rate_store[client_ip]
        _rate_store[client_ip] = [now]
        return False

    # Guard against unbounded growth of unique IPs
    if len(_rate_store) > _RATE_STORE_MAX_SIZE:
        oldest_ips = sorted(
            _rate_store,
            key=lambda k: _rate_store[k][-1] if _rate_store[k] else 0.0,
        )[: len(_rate_store) // 2]
        for k in oldest_ips:
            del _rate_store[k]
        if client_ip not in _rate_store:
            _rate_store[client_ip] = [now]
            return False

    if len(timestamps) >= _RATE_LIMIT:
        return True

    timestamps.append(now)
    return False


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Request size limit
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                content_length_int = int(content_length)
            except ValueError:
                return Response(
                    content='{"detail":"Invalid Content-Length"}',
                    status_code=400,
                    media_type="application/json",
                )
        else:
            content_length_int = 0
        if content_length_int > MAX_REQUEST_BODY:
            return Response(
                content='{"detail":"Request body too large"}',
                status_code=413,
                media_type="application/json",
            )

        # Rate limiting (skip health check)
        if request.url.path != "/api/health":
            client_ip = request.client.host if request.client else "unknown"
            if _is_rate_limited(client_ip):
                return Response(
                    content='{"detail":"Rate limit exceeded"}',
                    status_code=429,
                    media_type="application/json",
                    headers={"Retry-After": str(int(_RATE_WINDOW))},
                )

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except AppError as e:
            logger.warning(
                "%s %s -> %d: %s",
                request.method,
                request.url.path,
                e.status_code,
                e.message,
            )
            return Response(
                content=json.dumps({"detail": e.message}),
                status_code=e.status_code,
                media_type="application/json",
            )
        except Exception:
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            return Response(
                content='{"detail":"Internal server error"}',
                status_code=500,
                media_type="application/json",
            )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
