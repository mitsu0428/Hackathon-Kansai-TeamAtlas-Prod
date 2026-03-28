from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from src.domain.alert import Alert
from src.domain.anomaly import AnomalyResult
from src.domain.score import ScoreEntry
from src.infra.audio_loader import SUPPORTED_FORMATS, load_audio
from src.infra.config import settings
from src.use_case.detect_anomaly import detect_anomaly_async
from src.use_case.generate_intent import generate_intent

if TYPE_CHECKING:
    from src.domain.ports import EmbeddingPort, IndexPort, LLMPort

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Module-level detection context (set once via init_detection_context)
# ------------------------------------------------------------------


class _DetectionContext:
    def __init__(self) -> None:
        self.sensors: list = []
        self.alerts: list = []
        self.score_history: list = []
        self.clap: EmbeddingPort | None = None
        self.indices: dict[str, IndexPort] = {}
        self.index: IndexPort | None = None
        self.llm: LLMPort | None = None
        self.lock: asyncio.Lock | None = None


_ctx = _DetectionContext()


def init_detection_context(
    sensors: list,
    alerts: list,
    score_history: list,
    clap: EmbeddingPort,
    indices: dict[str, IndexPort],
    index: IndexPort,
    llm: LLMPort,
    lock: asyncio.Lock | None = None,
) -> None:
    """Initialise the module-level context used by the loop."""
    _ctx.sensors = sensors
    _ctx.alerts = alerts
    _ctx.score_history = score_history
    _ctx.clap = clap
    _ctx.indices = indices
    _ctx.index = index
    _ctx.llm = llm
    _ctx.lock = lock or asyncio.Lock()


def _get_index_for_sensor(sensor_id: str) -> IndexPort | None:
    """Get the best available index for a sensor."""
    idx = _ctx.indices.get(sensor_id)
    if idx is not None and idx.is_built:
        return idx
    idx = _ctx.indices.get("default")
    if idx is not None and idx.is_built:
        return idx
    if _ctx.index is not None and _ctx.index.is_built:
        return _ctx.index
    return None


# ------------------------------------------------------------------
# Sensor cursor tracking
# ------------------------------------------------------------------
_sensor_cursors: dict[str, int] = {}

MAX_SCORE_HISTORY = 1000
MAX_ALERTS = 100

_SIMULATION_BASE: dict[str, float] = {
    "urban": 0.20,
    "indoor": 0.10,
    "park": 0.25,
}


def _simulate_score(sensor_id: str) -> float:
    """Generate a simulated distance score for demo."""
    b = _SIMULATION_BASE.get(sensor_id, 0.15)
    return max(0.0, b + random.gauss(0, 0.05))


def _get_audio_files(sensor_id: str) -> list[Path]:
    """Get sorted audio files for a sensor."""
    sensor_dir = settings.streams_dir / sensor_id
    if not sensor_dir.exists():
        return []
    return sorted(p for p in sensor_dir.iterdir() if p.suffix.lower() in SUPPORTED_FORMATS)


async def _next_audio_for_sensor(sensor_id: str) -> Path | None:
    """Get next audio file for sensor, cycling."""
    files = _get_audio_files(sensor_id)
    if not files:
        return None
    async with _ctx.lock:
        cursor = _sensor_cursors.get(sensor_id, 0)
        if cursor >= len(files):
            cursor = 0
        _sensor_cursors[sensor_id] = cursor + 1
    return files[cursor]


async def _record_result(
    result: AnomalyResult,
    sensor_id: str,
) -> None:
    """Record score to history and trim if needed."""
    entry = ScoreEntry(
        sensor_id=sensor_id,
        timestamp=result.timestamp.isoformat(),
        distance=result.distance,
    )
    async with _ctx.lock:
        _ctx.score_history.append(entry)
        if len(_ctx.score_history) > MAX_SCORE_HISTORY:
            del _ctx.score_history[: len(_ctx.score_history) - MAX_SCORE_HISTORY]


async def _create_alert_if_anomaly(
    result: AnomalyResult,
    sensor_id: str,
) -> None:
    """Create an alert when an anomaly is detected."""
    if not result.is_anomaly:
        return

    intent = None
    if _ctx.llm is not None and _ctx.llm.is_available:
        try:
            sensor = next(
                (s for s in _ctx.sensors if s.sensor_id == sensor_id),
                None,
            )
            if sensor:
                intent = await generate_intent(result, sensor, _ctx.llm)
        except Exception as e:
            logger.warning("Intent generation failed: %s", e)

    alert = Alert(
        alert_id=str(uuid4()),
        sensor_id=sensor_id,
        timestamp=result.timestamp,
        anomaly=result,
        intent=intent,
    )
    async with _ctx.lock:
        _ctx.alerts.append(alert)
        if len(_ctx.alerts) > MAX_ALERTS:
            del _ctx.alerts[: len(_ctx.alerts) - MAX_ALERTS]


async def _detect_one_sensor(sensor_id: str) -> None:
    """Run detection for a single sensor."""
    index = _get_index_for_sensor(sensor_id)
    models_ready = _ctx.clap is not None and _ctx.clap.is_loaded and index is not None

    audio_path = await _next_audio_for_sensor(sensor_id)

    if not models_ready or audio_path is None:
        logger.debug(
            "Simulation mode for %s (models_ready=%s, audio=%s)",
            sensor_id,
            models_ready,
            audio_path,
        )
        now = datetime.now(tz=timezone.utc)
        distance = _simulate_score(sensor_id)
        threshold = settings.ANOMALY_THRESHOLD
        result = AnomalyResult(
            sensor_id=sensor_id,
            timestamp=now,
            distance=distance,
            is_anomaly=distance >= threshold,
            threshold=threshold,
        )
        await _record_result(result, sensor_id)
        await _create_alert_if_anomaly(result, sensor_id)
        return

    try:
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(None, load_audio, audio_path)
    except Exception as e:
        logger.warning(
            "Failed to load audio for %s: %s",
            sensor_id,
            e,
        )
        return

    try:
        result = await detect_anomaly_async(
            audio=audio,
            sensor_id=sensor_id,
            clap=_ctx.clap,
            index=index,
            threshold=settings.ANOMALY_THRESHOLD,
        )
    except Exception as e:
        logger.warning(
            "Detection failed for %s: %s",
            sensor_id,
            e,
        )
        return

    await _record_result(result, sensor_id)
    await _create_alert_if_anomaly(result, sensor_id)


async def run_detection_loop() -> None:
    """Background task: detect on all sensors periodically."""
    logger.info(
        "Detection loop started (interval=%ds)",
        settings.DETECTION_INTERVAL_SEC,
    )
    try:
        # Wait for server startup to complete before first detection
        await asyncio.sleep(2)
        while True:
            for sensor in _ctx.sensors:
                try:
                    await _detect_one_sensor(sensor.sensor_id)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(
                        "Detection loop error for %s: %s",
                        sensor.sensor_id,
                        e,
                    )
            await asyncio.sleep(settings.DETECTION_INTERVAL_SEC)
    except asyncio.CancelledError:
        logger.info("Detection loop stopped")
