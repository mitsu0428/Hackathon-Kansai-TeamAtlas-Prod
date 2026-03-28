from datetime import datetime

from pydantic import BaseModel


class Baseline(BaseModel):
    sensor_id: str
    created_at: datetime
    sample_count: int
    index_path: str  # faiss index file path
