from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.domain.anomaly import AnomalyResult
from src.domain.intent import Intent
from src.domain.sensor import Sensor

if TYPE_CHECKING:
    from src.domain.ports import LLMPort

logger = logging.getLogger(__name__)


async def generate_intent(
    anomaly: AnomalyResult,
    sensor: Sensor,
    llm: LLMPort,
    context: dict | None = None,
) -> Intent:
    """Generate an interpretation of an anomaly detection result."""
    return await llm.interpret(anomaly, sensor, context)
