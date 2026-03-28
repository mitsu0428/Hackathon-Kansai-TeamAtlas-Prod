from datetime import datetime

from pydantic import BaseModel


class SensorStatus(BaseModel):
    sensor_id: str
    name: str
    location: str
    is_active: bool
    last_checked: datetime | None = None
    current_distance: float = 0.0
    is_anomaly: bool = False
