from pydantic import BaseModel


class Sensor(BaseModel):
    sensor_id: str  # e.g. "urban", "indoor", "park"
    name: str  # 表示名
    location: str  # 設置場所の説明
