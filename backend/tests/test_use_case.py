"""Unit tests for use-case layer (detect_anomaly, get_status)."""

from datetime import datetime, timezone

import numpy as np
import pytest

from src.domain.score import ScoreEntry
from src.domain.sensor import Sensor
from src.use_case.detect_anomaly import detect_anomaly
from src.use_case.get_status import get_status


# ---------------------------------------------------------------------------
# Mock implementations for EmbeddingPort / IndexPort
# ---------------------------------------------------------------------------


class MockEmbedding:
    is_loaded = True

    def embed(self, audio, sr=48000):
        return np.zeros((1, 512), dtype=np.float32)

    def embed_batch(self, audios, sr=48000):
        return np.zeros((len(audios), 512), dtype=np.float32)


class MockIndexNormal:
    """Returns low distances → normal detection."""

    is_built = True
    ntotal = 100

    def search(self, vector, k=5):
        distances = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        indices = np.array([[0, 1, 2, 3, 4]])
        metadata = [{"labels": ["Traffic noise"]} for _ in range(5)]
        return distances, indices, metadata

    def get_all_labels(self):
        return ["Traffic noise", "Engine"]

    def build(self, vectors):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def set_metadata(self, metadata):
        pass


class MockIndexAnomaly:
    """Returns high distances → anomaly detection."""

    is_built = True
    ntotal = 100

    def search(self, vector, k=5):
        # Inner Product: low similarity = anomaly (far from baseline)
        distances = np.array([[0.3, 0.2, 0.1, 0.15, 0.25]])
        indices = np.array([[0, 1, 2, 3, 4]])
        metadata = [{"labels": ["Siren"]} for _ in range(5)]
        return distances, indices, metadata

    def get_all_labels(self):
        return ["Traffic noise", "Engine", "Siren"]

    def build(self, vectors):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def set_metadata(self, metadata):
        pass


# ---------------------------------------------------------------------------
# detect_anomaly tests
# ---------------------------------------------------------------------------


class TestDetectAnomaly:
    def test_detect_anomaly_normal(self):
        """Distance below threshold should yield is_anomaly=False."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="urban",
            clap=MockEmbedding(),
            index=MockIndexNormal(),
            threshold=0.6,
        )
        assert result.is_anomaly is False
        assert result.distance < 0.6
        assert result.sensor_id == "urban"
        assert result.threshold == 0.6

    def test_detect_anomaly_anomaly(self):
        """Distance above threshold should yield is_anomaly=True."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="park",
            clap=MockEmbedding(),
            index=MockIndexAnomaly(),
            threshold=0.6,
        )
        assert result.is_anomaly is True
        assert result.distance >= 0.6
        assert result.sensor_id == "park"

    def test_detect_anomaly_labels(self):
        """Verify matched_labels and baseline_categories are populated."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="indoor",
            clap=MockEmbedding(),
            index=MockIndexAnomaly(),
            threshold=0.6,
        )
        assert "Siren" in result.matched_labels
        assert "Traffic noise" in result.baseline_categories
        assert "Engine" in result.baseline_categories
        assert "Siren" in result.baseline_categories

    def test_detect_anomaly_normal_labels(self):
        """Normal result should still carry label information."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="urban",
            clap=MockEmbedding(),
            index=MockIndexNormal(),
            threshold=0.6,
        )
        assert "Traffic noise" in result.matched_labels
        assert result.baseline_categories == ["Traffic noise", "Engine"]

    def test_detect_anomaly_timestamp_present(self):
        """Result should have a UTC timestamp."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="urban",
            clap=MockEmbedding(),
            index=MockIndexNormal(),
            threshold=0.6,
        )
        assert result.timestamp is not None
        assert result.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# get_status tests
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_get_status_with_scores(self):
        """Sensors with score entries should be active with correct anomaly flag."""
        sensors = [
            Sensor(sensor_id="urban", name="Urban Sensor", location="City center"),
            Sensor(sensor_id="indoor", name="Indoor Sensor", location="Office 3F"),
        ]
        score_history = [
            ScoreEntry(
                sensor_id="urban",
                timestamp="2025-01-01T12:00:00+00:00",
                distance=0.8,
            ),
            ScoreEntry(
                sensor_id="indoor",
                timestamp="2025-01-01T12:00:00+00:00",
                distance=0.3,
            ),
        ]
        statuses = get_status(sensors, score_history, threshold=0.6)

        assert len(statuses) == 2

        urban_status = next(s for s in statuses if s.sensor_id == "urban")
        assert urban_status.is_active is True
        assert urban_status.is_anomaly is True
        assert urban_status.current_distance == 0.8

        indoor_status = next(s for s in statuses if s.sensor_id == "indoor")
        assert indoor_status.is_active is True
        assert indoor_status.is_anomaly is False
        assert indoor_status.current_distance == 0.3

    def test_get_status_empty(self):
        """No score entries → all sensors inactive with defaults."""
        sensors = [
            Sensor(sensor_id="urban", name="Urban Sensor", location="City center"),
            Sensor(sensor_id="park", name="Park Sensor", location="Central park"),
        ]
        statuses = get_status(sensors, score_history=[], threshold=0.6)

        assert len(statuses) == 2
        for status in statuses:
            assert status.is_active is False
            assert status.is_anomaly is False
            assert status.current_distance == 0.0
            assert status.last_checked is None

    def test_get_status_latest_entry_wins(self):
        """When multiple entries exist for a sensor, the latest one is used."""
        sensors = [
            Sensor(sensor_id="urban", name="Urban Sensor", location="City center"),
        ]
        score_history = [
            ScoreEntry(
                sensor_id="urban",
                timestamp="2025-01-01T10:00:00+00:00",
                distance=0.2,
            ),
            ScoreEntry(
                sensor_id="urban",
                timestamp="2025-01-01T14:00:00+00:00",
                distance=0.9,
            ),
            ScoreEntry(
                sensor_id="urban",
                timestamp="2025-01-01T12:00:00+00:00",
                distance=0.4,
            ),
        ]
        statuses = get_status(sensors, score_history, threshold=0.6)
        assert len(statuses) == 1
        assert statuses[0].current_distance == 0.9
        assert statuses[0].is_anomaly is True

    def test_get_status_mixed_active_inactive(self):
        """Some sensors have scores, others do not."""
        sensors = [
            Sensor(sensor_id="urban", name="Urban Sensor", location="City center"),
            Sensor(sensor_id="park", name="Park Sensor", location="Central park"),
        ]
        score_history = [
            ScoreEntry(
                sensor_id="urban",
                timestamp="2025-01-01T12:00:00+00:00",
                distance=0.5,
            ),
        ]
        statuses = get_status(sensors, score_history, threshold=0.6)

        urban = next(s for s in statuses if s.sensor_id == "urban")
        park = next(s for s in statuses if s.sensor_id == "park")

        assert urban.is_active is True
        assert urban.is_anomaly is False
        assert park.is_active is False
