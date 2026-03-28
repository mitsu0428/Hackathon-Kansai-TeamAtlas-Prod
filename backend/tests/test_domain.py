"""Unit tests for domain models (pure, no mocks needed)."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.domain.anomaly import AnomalyResult
from src.domain.alert import Alert
from src.domain.baseline import Baseline
from src.domain.categories import CATEGORY_SENSOR_MAP, LABEL_TO_SENSOR, build_label_to_sensor
from src.domain.intent import Intent
from src.domain.score import ScoreEntry
from src.domain.sensor import Sensor


# ---------------------------------------------------------------------------
# AnomalyResult
# ---------------------------------------------------------------------------


class TestAnomalyResult:
    def test_creation_anomaly(self, sample_anomaly_result: AnomalyResult):
        assert sample_anomaly_result.is_anomaly is True
        assert sample_anomaly_result.distance >= sample_anomaly_result.threshold
        assert sample_anomaly_result.sensor_id == "urban"
        assert "Siren" in sample_anomaly_result.matched_labels

    def test_creation_normal(self, sample_normal_result: AnomalyResult):
        assert sample_normal_result.is_anomaly is False
        assert sample_normal_result.distance < sample_normal_result.threshold

    def test_at_threshold_boundary(self):
        """distance == threshold should be treated as anomaly per detect_anomaly logic (>=)."""
        result = AnomalyResult(
            sensor_id="indoor",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            distance=0.6,
            is_anomaly=True,  # distance >= threshold
            threshold=0.6,
        )
        assert result.is_anomaly is True

    def test_serialization_roundtrip(self, sample_anomaly_result: AnomalyResult):
        json_str = sample_anomaly_result.model_dump_json()
        restored = AnomalyResult.model_validate_json(json_str)
        assert restored == sample_anomaly_result

    def test_default_empty_labels(self):
        result = AnomalyResult(
            sensor_id="park",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            distance=0.1,
            is_anomaly=False,
            threshold=0.6,
        )
        assert result.matched_labels == []
        assert result.baseline_categories == []


# ---------------------------------------------------------------------------
# ScoreEntry
# ---------------------------------------------------------------------------


class TestScoreEntry:
    def test_creation(self, sample_score_entry: ScoreEntry):
        assert sample_score_entry.sensor_id == "indoor"
        assert sample_score_entry.distance == 0.45

    def test_serialization_roundtrip(self, sample_score_entry: ScoreEntry):
        json_str = sample_score_entry.model_dump_json()
        restored = ScoreEntry.model_validate_json(json_str)
        assert restored == sample_score_entry


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------


class TestAlert:
    def test_alert_without_intent(self, sample_alert: Alert):
        assert sample_alert.alert_id == "alert-001"
        assert sample_alert.intent is None
        assert sample_alert.anomaly.is_anomaly is True

    def test_alert_with_intent(self, sample_alert_with_intent: Alert):
        assert sample_alert_with_intent.intent is not None
        assert sample_alert_with_intent.intent.urgency == "high"

    def test_serialization_roundtrip(self, sample_alert_with_intent: Alert):
        json_str = sample_alert_with_intent.model_dump_json()
        restored = Alert.model_validate_json(json_str)
        assert restored == sample_alert_with_intent


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------


class TestSensor:
    def test_creation(self):
        sensor = Sensor(sensor_id="urban", name="Urban Sensor", location="City center")
        assert sensor.sensor_id == "urban"
        assert sensor.name == "Urban Sensor"

    def test_serialization(self):
        sensor = Sensor(sensor_id="park", name="Park Sensor", location="Central park")
        data = sensor.model_dump()
        assert data["sensor_id"] == "park"


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------


class TestIntent:
    def test_creation(self, sample_intent: Intent):
        assert sample_intent.urgency == "high"
        assert sample_intent.sensor_id == "urban"

    def test_urgency_values(self):
        """Urgency must be one of low/medium/high/critical."""
        for urgency in ("low", "medium", "high", "critical"):
            intent = Intent(
                sensor_id="test",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                judgment="test",
                recommendation="test",
                urgency=urgency,
                supplement="test",
            )
            assert intent.urgency == urgency


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_creation(self):
        bl = Baseline(
            sensor_id="urban",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            sample_count=100,
            index_path="/data/index/urban_baseline.faiss",
        )
        assert bl.sample_count == 100
        assert bl.sensor_id == "urban"


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


class TestCategories:
    def test_expected_keys(self):
        assert "urban" in CATEGORY_SENSOR_MAP
        assert "indoor" in CATEGORY_SENSOR_MAP
        assert "park" in CATEGORY_SENSOR_MAP

    def test_each_key_has_labels(self):
        for key, labels in CATEGORY_SENSOR_MAP.items():
            assert len(labels) > 0, f"Key {key!r} has no labels"

    def test_label_to_sensor_reverse_mapping(self):
        mapping = build_label_to_sensor()
        assert mapping["siren"] == "urban"
        assert mapping["bird"] == "park"
        assert mapping["air conditioning"] == "indoor"

    def test_label_to_sensor_module_level(self):
        assert isinstance(LABEL_TO_SENSOR, dict)
        assert len(LABEL_TO_SENSOR) > 0


# ---------------------------------------------------------------------------
# Domain validation edge cases
# ---------------------------------------------------------------------------


class TestDomainValidation:
    def test_anomaly_result_negative_distance(self):
        """Negative distance is accepted by Pydantic (no constraint defined)."""
        result = AnomalyResult(
            sensor_id="test",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            distance=-1.0,
            is_anomaly=False,
            threshold=0.6,
        )
        assert result.distance == -1.0

    def test_intent_invalid_urgency(self):
        """Urgency must be one of the Literal values; invalid value raises ValidationError."""
        with pytest.raises(ValidationError):
            Intent(
                sensor_id="test",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                judgment="test",
                recommendation="test",
                urgency="extreme",  # not in Literal["low","medium","high","critical"]
                supplement="test",
            )
