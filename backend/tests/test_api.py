"""Integration tests for API endpoints using a lightweight test app."""

import pytest
from httpx import AsyncClient

from src.api.middleware import _RATE_LIMIT, _rate_store


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_health_returns_200(self, client: AsyncClient):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "components" in body
        assert set(body["components"].keys()) == {"clap", "faiss", "llm"}

    async def test_health_components_false_in_test_mode(self, client: AsyncClient):
        resp = await client.get("/api/health")
        components = resp.json()["components"]
        assert components["clap"] is False
        assert components["faiss"] is False
        assert components["llm"] is False


# ---------------------------------------------------------------------------
# Sensors / Status
# ---------------------------------------------------------------------------


class TestSensorsStatus:
    async def test_status_returns_200(self, client: AsyncClient):
        resp = await client.get("/api/sensors/status")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_scores_returns_200(self, client: AsyncClient):
        resp = await client.get("/api/sensors/scores")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


class TestAlerts:
    async def test_alerts_returns_200(self, client: AsyncClient):
        resp = await client.get("/api/alerts")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Request size limit
# ---------------------------------------------------------------------------


class TestRequestSizeLimit:
    async def test_oversized_content_length_returns_413(self, client: AsyncClient):
        resp = await client.post(
            "/api/health",
            headers={"content-length": "999999999"},
            content=b"",
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    async def test_rate_limit_returns_429_with_retry_after(self, client: AsyncClient):
        """Exhaust the rate limit, then verify 429 + Retry-After header."""
        # We target a non-health endpoint (health is exempt from rate limiting)
        _rate_store.clear()
        for _ in range(_RATE_LIMIT):
            await client.get("/api/alerts")

        resp = await client.get("/api/alerts")
        assert resp.status_code == 429
        assert "retry-after" in resp.headers
        body = resp.json()
        assert "rate limit" in body["detail"].lower()
