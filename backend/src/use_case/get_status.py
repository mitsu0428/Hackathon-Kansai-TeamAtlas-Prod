import logging
from datetime import datetime

from src.domain.score import ScoreEntry
from src.domain.sensor import Sensor
from src.domain.status import SensorStatus

logger = logging.getLogger(__name__)


def get_status(
    sensors: list[Sensor],
    score_history: list[ScoreEntry],
    threshold: float,
) -> list[SensorStatus]:
    """Build current status for all sensors from score history."""
    latest: dict[str, ScoreEntry] = {}
    for entry in score_history:
        if entry.sensor_id not in latest or entry.timestamp > latest[entry.sensor_id].timestamp:
            latest[entry.sensor_id] = entry

    statuses = []
    for sensor in sensors:
        entry = latest.get(sensor.sensor_id)
        if entry:
            ts = entry.timestamp
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            statuses.append(
                SensorStatus(
                    sensor_id=sensor.sensor_id,
                    name=sensor.name,
                    location=sensor.location,
                    is_active=True,
                    last_checked=ts,
                    current_distance=entry.distance,
                    is_anomaly=entry.distance >= threshold,
                )
            )
        else:
            statuses.append(
                SensorStatus(
                    sensor_id=sensor.sensor_id,
                    name=sensor.name,
                    location=sensor.location,
                    is_active=False,
                )
            )
    return statuses
