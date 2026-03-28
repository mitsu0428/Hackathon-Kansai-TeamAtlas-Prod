"""Test that 'disappeared sound' detection works correctly.

When a normal sound (e.g., AC) disappears and is replaced by silence,
the system should:
1. Detect high distance (anomaly)
2. Identify which sounds are missing from baseline
3. Provide enough context for LLM to explain what happened
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from src.domain.anomaly import AnomalyResult
from src.domain.sensor import Sensor
from src.infra.llm_client import _build_user_prompt
from src.use_case.detect_anomaly import detect_anomaly


# ---------------------------------------------------------------------------
# Mock implementations for disappeared-sound scenario
# ---------------------------------------------------------------------------


class MockSilenceEmbedding:
    """Simulates embedding model producing vectors for silence audio."""

    is_loaded = True

    def embed(self, audio, sr=48000):
        return np.random.randn(1, 512).astype(np.float32)

    def embed_batch(self, audios, sr=48000):
        return np.random.randn(len(audios), 512).astype(np.float32)


class MockIndexWithACBaseline:
    """Baseline contains indoor sounds (AC, fan, etc.).

    When silence is searched, distances are high because silence
    is far from all baseline sounds.
    """

    is_built = True
    ntotal = 100

    def search(self, vector, k=5):
        # Inner Product: low similarity = silence far from baseline
        distances = np.array([[0.25, 0.22, 0.18, 0.15, 0.10]])
        indices = np.array([[0, 1, 2, 3, 4]])
        metadata = [
            {"labels": ["Air conditioning"]},
            {"labels": ["Mechanical fan"]},
            {"labels": ["Air conditioning"]},
            {"labels": ["Footsteps"]},
            {"labels": ["Speech"]},
        ]
        return distances, indices, metadata

    def get_all_labels(self):
        return [
            "Air conditioning",
            "Door",
            "Footsteps",
            "Mechanical fan",
            "Speech",
        ]

    def build(self, vectors):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def set_metadata(self, metadata):
        pass


class MockIndexNormalIndoor:
    """Baseline contains indoor sounds and returns LOW distances
    when normal indoor audio (with AC/fan) is present.
    """

    is_built = True
    ntotal = 100

    def search(self, vector, k=5):
        # Inner Product: high similarity = normal audio close to baseline
        distances = np.array([[0.90, 0.85, 0.80, 0.78, 0.72]])
        indices = np.array([[0, 1, 2, 3, 4]])
        metadata = [
            {"labels": ["Air conditioning"]},
            {"labels": ["Air conditioning"]},
            {"labels": ["Mechanical fan"]},
            {"labels": ["Air conditioning"]},
            {"labels": ["Mechanical fan"]},
        ]
        return distances, indices, metadata

    def get_all_labels(self):
        return [
            "Air conditioning",
            "Door",
            "Footsteps",
            "Mechanical fan",
            "Speech",
        ]

    def build(self, vectors):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def set_metadata(self, metadata):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

THRESHOLD = 0.55


class TestSilenceProducesHighDistance:
    """When silence replaces normal indoor sounds, anomaly is detected."""

    def test_silence_triggers_anomaly(self):
        """Silence audio against AC/fan baseline should yield is_anomaly=True."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="indoor",
            clap=MockSilenceEmbedding(),
            index=MockIndexWithACBaseline(),
            threshold=THRESHOLD,
        )
        assert result.is_anomaly is True

    def test_silence_distance_exceeds_threshold(self):
        """Distance score should be well above the threshold."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="indoor",
            clap=MockSilenceEmbedding(),
            index=MockIndexWithACBaseline(),
            threshold=THRESHOLD,
        )
        assert result.distance >= THRESHOLD
        # 1 - max(0.25, 0.22, 0.18, 0.15, 0.10) = 1 - 0.25 = 0.75
        assert abs(result.distance - 0.75) < 0.01

    def test_matched_labels_contain_baseline_sounds(self):
        """Nearest neighbors should reference AC/fan sounds from baseline."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="indoor",
            clap=MockSilenceEmbedding(),
            index=MockIndexWithACBaseline(),
            threshold=THRESHOLD,
        )
        assert "Air conditioning" in result.matched_labels
        assert "Mechanical fan" in result.matched_labels

    def test_baseline_categories_contain_all_indoor_sounds(self):
        """Baseline categories should list all known indoor sound types."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="indoor",
            clap=MockSilenceEmbedding(),
            index=MockIndexWithACBaseline(),
            threshold=THRESHOLD,
        )
        expected = {"Air conditioning", "Door", "Footsteps", "Mechanical fan", "Speech"}
        assert set(result.baseline_categories) == expected


class TestMissingSoundsComputed:
    """After anomaly detection, missing sounds can be derived from the result."""

    def test_missing_sounds_identified(self):
        """Sounds in baseline but NOT in matched_labels are 'missing'."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="indoor",
            clap=MockSilenceEmbedding(),
            index=MockIndexWithACBaseline(),
            threshold=THRESHOLD,
        )
        missing = set(result.baseline_categories) - set(result.matched_labels)
        # "Door" is in baseline but not in any matched neighbor metadata
        assert "Door" in missing

    def test_matched_sounds_not_in_missing(self):
        """Sounds that appear in matched_labels should NOT be in missing set."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="indoor",
            clap=MockSilenceEmbedding(),
            index=MockIndexWithACBaseline(),
            threshold=THRESHOLD,
        )
        missing = set(result.baseline_categories) - set(result.matched_labels)
        assert "Air conditioning" not in missing
        assert "Mechanical fan" not in missing
        assert "Footsteps" not in missing
        assert "Speech" not in missing


class TestLLMPromptIncludesMissingSounds:
    """The LLM prompt should explicitly list sounds that have disappeared."""

    def test_prompt_contains_potentially_missing(self):
        """Prompt should contain 'Potentially missing:' with AC and fan."""
        anomaly = AnomalyResult(
            sensor_id="indoor",
            timestamp=datetime(2025, 7, 15, 14, 0, 0, tzinfo=timezone.utc),
            distance=0.82,
            is_anomaly=True,
            threshold=THRESHOLD,
            matched_labels=["Speech"],
            baseline_categories=["Air conditioning", "Mechanical fan", "Speech"],
        )
        sensor = Sensor(
            sensor_id="indoor",
            name="Indoor Sensor",
            location="Office 3F",
        )
        prompt = _build_user_prompt(anomaly, sensor, {})

        assert "Potentially missing:" in prompt
        assert "Air conditioning" in prompt
        assert "Mechanical fan" in prompt

    def test_prompt_contains_baseline_sounds(self):
        """Prompt should list all baseline sounds."""
        anomaly = AnomalyResult(
            sensor_id="indoor",
            timestamp=datetime(2025, 7, 15, 14, 0, 0, tzinfo=timezone.utc),
            distance=0.82,
            is_anomaly=True,
            threshold=THRESHOLD,
            matched_labels=["Speech"],
            baseline_categories=["Air conditioning", "Mechanical fan", "Speech"],
        )
        sensor = Sensor(
            sensor_id="indoor",
            name="Indoor Sensor",
            location="Office 3F",
        )
        prompt = _build_user_prompt(anomaly, sensor, {})

        assert "Baseline sounds:" in prompt
        assert "Air conditioning" in prompt
        assert "Mechanical fan" in prompt
        assert "Speech" in prompt

    def test_prompt_contains_matched_sounds(self):
        """Prompt should list matched (detected) sounds."""
        anomaly = AnomalyResult(
            sensor_id="indoor",
            timestamp=datetime(2025, 7, 15, 14, 0, 0, tzinfo=timezone.utc),
            distance=0.82,
            is_anomaly=True,
            threshold=THRESHOLD,
            matched_labels=["Speech"],
            baseline_categories=["Air conditioning", "Mechanical fan", "Speech"],
        )
        sensor = Sensor(
            sensor_id="indoor",
            name="Indoor Sensor",
            location="Office 3F",
        )
        prompt = _build_user_prompt(anomaly, sensor, {})

        assert "Matched sounds:" in prompt

    def test_prompt_missing_list_is_sorted(self):
        """The 'Potentially missing' list should be sorted alphabetically."""
        anomaly = AnomalyResult(
            sensor_id="indoor",
            timestamp=datetime(2025, 7, 15, 14, 0, 0, tzinfo=timezone.utc),
            distance=0.82,
            is_anomaly=True,
            threshold=THRESHOLD,
            matched_labels=["Speech"],
            baseline_categories=["Air conditioning", "Mechanical fan", "Speech"],
        )
        sensor = Sensor(
            sensor_id="indoor",
            name="Indoor Sensor",
            location="Office 3F",
        )
        prompt = _build_user_prompt(anomaly, sensor, {})

        # Extract the "Potentially missing" line
        for line in prompt.split("\n"):
            if "Potentially missing:" in line:
                # "Air conditioning" should come before "Mechanical fan"
                ac_pos = line.index("Air conditioning")
                fan_pos = line.index("Mechanical fan")
                assert ac_pos < fan_pos
                break
        else:
            pytest.fail("'Potentially missing:' line not found in prompt")

    def test_no_missing_when_all_matched(self):
        """If all baseline sounds are matched, no 'Potentially missing' line."""
        anomaly = AnomalyResult(
            sensor_id="indoor",
            timestamp=datetime(2025, 7, 15, 14, 0, 0, tzinfo=timezone.utc),
            distance=0.35,
            is_anomaly=False,
            threshold=THRESHOLD,
            matched_labels=["Air conditioning", "Mechanical fan", "Speech"],
            baseline_categories=["Air conditioning", "Mechanical fan", "Speech"],
        )
        sensor = Sensor(
            sensor_id="indoor",
            name="Indoor Sensor",
            location="Office 3F",
        )
        prompt = _build_user_prompt(anomaly, sensor, {})

        assert "Potentially missing:" not in prompt


class TestNormalSoundProducesLowDistance:
    """When normal indoor sounds are present, no anomaly should be detected."""

    def test_normal_indoor_is_not_anomaly(self):
        """Normal AC/fan audio against AC/fan baseline should not trigger."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="indoor",
            clap=MockSilenceEmbedding(),
            index=MockIndexNormalIndoor(),
            threshold=THRESHOLD,
        )
        assert result.is_anomaly is False

    def test_normal_indoor_low_distance(self):
        """Distance should be well below threshold."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="indoor",
            clap=MockSilenceEmbedding(),
            index=MockIndexNormalIndoor(),
            threshold=THRESHOLD,
        )
        assert result.distance < THRESHOLD
        # 1 - max(0.90, 0.85, 0.80, 0.78, 0.72) = 1 - 0.90 = 0.10
        assert abs(result.distance - 0.10) < 0.01

    def test_normal_indoor_matched_labels(self):
        """Matched labels should reflect typical indoor sounds."""
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_anomaly(
            audio=audio,
            sensor_id="indoor",
            clap=MockSilenceEmbedding(),
            index=MockIndexNormalIndoor(),
            threshold=THRESHOLD,
        )
        assert "Air conditioning" in result.matched_labels
        assert "Mechanical fan" in result.matched_labels
