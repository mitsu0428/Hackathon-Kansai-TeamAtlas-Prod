from fastapi import APIRouter, Query

from src.api.deps import SCORE_HISTORY, SENSORS, state_lock
from src.domain.score import ScoreEntry
from src.domain.status import SensorStatus
from src.infra.config import settings
from src.use_case.get_status import get_status

router = APIRouter(prefix="/api/sensors")


@router.get("/status", response_model=list[SensorStatus])
async def get_sensor_status() -> list[SensorStatus]:
    """Return current status of all sensors."""
    async with state_lock:
        history_snapshot = list(SCORE_HISTORY)
    return get_status(SENSORS, history_snapshot, settings.ANOMALY_THRESHOLD)


@router.get("/scores", response_model=list[ScoreEntry])
async def get_score_history(
    limit: int = Query(100, ge=1, le=1000),
    sensor_id: str | None = Query(
        None,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]*$",
    ),
) -> list[ScoreEntry]:
    """Return recent score history, optionally filtered."""
    async with state_lock:
        scores = list(SCORE_HISTORY)
    if sensor_id:
        scores = [s for s in scores if s.sensor_id == sensor_id]
    return scores[-limit:]
