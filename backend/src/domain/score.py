from pydantic import BaseModel


class ScoreEntry(BaseModel):
    sensor_id: str
    timestamp: str  # ISO format string
    distance: float
