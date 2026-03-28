from fastapi import APIRouter

from src.api.deps import ALERTS, state_lock
from src.domain.alert import Alert

router = APIRouter(prefix="/api")


@router.get("/alerts", response_model=list[Alert])
async def get_alerts() -> list[Alert]:
    """Return alerts in reverse chronological order."""
    async with state_lock:
        return list(reversed(ALERTS))
