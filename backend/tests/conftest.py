"""Shared fixtures for the test suite."""

from datetime import datetime, timezone
from unittest.mock import PropertyMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.domain.alert import Alert
from src.domain.anomaly import AnomalyResult
from src.domain.intent import Intent
from src.domain.score import ScoreEntry
from src.domain.sensor import Sensor


# ---------------------------------------------------------------------------
# Domain object fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_anomaly_result() -> AnomalyResult:
    return AnomalyResult(
        sensor_id="urban",
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        distance=0.85,
        is_anomaly=True,
        threshold=0.6,
        matched_labels=["Siren", "Car"],
        baseline_categories=["Traffic noise, roadway noise", "Siren"],
    )


@pytest.fixture
def sample_normal_result() -> AnomalyResult:
    return AnomalyResult(
        sensor_id="park",
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        distance=0.3,
        is_anomaly=False,
        threshold=0.6,
        matched_labels=["Bird"],
        baseline_categories=["Bird", "Wind"],
    )


@pytest.fixture
def sample_score_entry() -> ScoreEntry:
    return ScoreEntry(
        sensor_id="indoor",
        timestamp="2025-01-01T12:00:00+00:00",
        distance=0.45,
    )


@pytest.fixture
def sample_intent() -> Intent:
    return Intent(
        sensor_id="urban",
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        judgment="Unusual siren activity detected near intersection.",
        recommendation="Dispatch security patrol to verify source.",
        urgency="high",
        supplement="Multiple siren-like sounds detected within 5 minutes.",
    )


@pytest.fixture
def sample_alert(sample_anomaly_result: AnomalyResult) -> Alert:
    return Alert(
        alert_id="alert-001",
        sensor_id="urban",
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        anomaly=sample_anomaly_result,
        intent=None,
    )


@pytest.fixture
def sample_alert_with_intent(
    sample_anomaly_result: AnomalyResult,
    sample_intent: Intent,
) -> Alert:
    return Alert(
        alert_id="alert-002",
        sensor_id="urban",
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        anomaly=sample_anomaly_result,
        intent=sample_intent,
    )


@pytest.fixture
def sample_sensors() -> list[Sensor]:
    return [
        Sensor(sensor_id="urban", name="Urban Sensor", location="City center"),
        Sensor(sensor_id="indoor", name="Indoor Sensor", location="Office 3F"),
        Sensor(sensor_id="park", name="Park Sensor", location="Central park"),
    ]


# ---------------------------------------------------------------------------
# Test FastAPI app (no ML models)
# ---------------------------------------------------------------------------


def _create_test_app():
    """Build a minimal FastAPI app that does not load ML models."""
    from fastapi import FastAPI

    from src.api.middleware import RequestLoggingMiddleware
    from src.api.routes import alerts, status

    app = FastAPI(title="Test App")
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(status.router)
    app.include_router(alerts.router)

    # Stub health endpoint matching main app structure
    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "components": {
                "clap": False,
                "faiss": False,
                "llm": False,
            },
        }

    return app


@pytest.fixture
async def client():
    """Async test client backed by the lightweight test app."""
    from src.api.middleware import _rate_store

    # Clear rate-limit state between tests
    _rate_store.clear()

    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
