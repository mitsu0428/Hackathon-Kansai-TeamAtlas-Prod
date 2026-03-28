import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Path

from src.api.deps import faiss_index, faiss_indices
from src.domain.baseline import Baseline
from src.infra.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/baseline/{sensor_id}", response_model=Baseline, response_model_exclude={"index_path"})
async def get_baseline(
    sensor_id: str = Path(
        ...,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",
    ),
) -> Baseline:
    """Return baseline information for the specified sensor."""
    idx_path = settings.index_dir / f"{sensor_id}_baseline.faiss"

    idx = faiss_indices.get(sensor_id)
    if idx is None:
        idx = faiss_index

    sample_count = idx.ntotal if idx.is_built else 0

    try:
        if idx_path.exists():
            stat = idx_path.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        else:
            created_at = datetime.now(tz=timezone.utc)
    except OSError as e:
        logger.warning("Failed to stat index file %s: %s", idx_path, e)
        created_at = datetime.now(tz=timezone.utc)

    return Baseline(
        sensor_id=sensor_id,
        created_at=created_at,
        sample_count=sample_count,
        index_path=str(idx_path),
    )
