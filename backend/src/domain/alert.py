from datetime import datetime

from pydantic import BaseModel

from src.domain.anomaly import AnomalyResult
from src.domain.intent import Intent


class Alert(BaseModel):
    alert_id: str
    sensor_id: str
    timestamp: datetime
    anomaly: AnomalyResult
    intent: Intent | None = None  # None if LLM disabled
