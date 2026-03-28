from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Urgency = Literal["low", "medium", "high", "critical"]


class Intent(BaseModel):
    sensor_id: str
    timestamp: datetime
    judgment: str
    recommendation: str
    urgency: Urgency
    supplement: str
