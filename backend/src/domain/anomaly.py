from datetime import datetime

from pydantic import BaseModel, Field


class AnomalyResult(BaseModel):
    sensor_id: str
    timestamp: datetime
    distance: float  # baseline distance (larger = more anomalous)
    is_anomaly: bool
    threshold: float
    matched_labels: list[str] = Field(default_factory=list)
    baseline_categories: list[str] = Field(default_factory=list)
