from __future__ import annotations

import asyncio
import logging

from src.domain.alert import Alert  # noqa: F811, TCH001
from src.domain.score import ScoreEntry
from src.domain.sensor import Sensor
from src.infra.clap_model import ClapModel
from src.infra.config import settings
from src.infra.faiss_index import FaissIndex
from src.infra.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SENSOR_JP: dict[str, tuple[str, str]] = {
    "urban": ("都市センサー", "市街地 交差点付近"),
    "indoor": ("室内センサー", "オフィスビル 3F"),
    "park": ("公園センサー", "中央公園 南入口"),
}

_DEFAULT_SENSORS: list[Sensor] = [
    Sensor(
        sensor_id="urban",
        name="都市センサー",
        location="市街地 交差点付近",
    ),
    Sensor(
        sensor_id="indoor",
        name="室内センサー",
        location="オフィスビル 3F",
    ),
    Sensor(
        sensor_id="park",
        name="公園センサー",
        location="中央公園 南入口",
    ),
]


def _discover_sensors() -> list[Sensor]:
    """Auto-discover sensors from data/streams/ directories.

    Falls back to defaults if no streams directories exist.
    """
    streams_dir = settings.streams_dir
    if not streams_dir.exists():
        logger.info(
            "streams_dir %s not found, using defaults",
            streams_dir,
        )
        return list(_DEFAULT_SENSORS)
    dirs = sorted(d.name for d in streams_dir.iterdir() if d.is_dir())
    if not dirs:
        logger.info("No subdirs in %s, using defaults", streams_dir)
        return list(_DEFAULT_SENSORS)
    sensors: list[Sensor] = []
    for d in dirs:
        jp = _SENSOR_JP.get(d)
        if jp:
            name, location = jp
        else:
            pretty = d.replace("_", " ").title()
            name = pretty + " センサー"
            location = pretty + " エリア"
        sensors.append(
            Sensor(
                sensor_id=d,
                name=name,
                location=location,
            )
        )
    logger.info("Discovered %d sensors from %s", len(sensors), streams_dir)
    return sensors


SENSORS: list[Sensor] = _discover_sensors()

ALERTS: list["Alert"] = []
SCORE_HISTORY: list[ScoreEntry] = []

# Lock for concurrent access to ALERTS and SCORE_HISTORY
state_lock = asyncio.Lock()

# Singleton instances (lifespan)
clap_model = ClapModel()
faiss_index = FaissIndex(dimension=512)  # fallback default index
faiss_indices: dict[str, FaissIndex] = {}  # per-sensor indices
llm_client = LLMClient()


def save_state() -> None:
    """Write ALERTS and SCORE_HISTORY to JSONL files for persistence."""
    state_dir = settings.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / "alerts.jsonl", "w") as f:
        for a in ALERTS:
            f.write(a.model_dump_json() + "\n")
    with open(state_dir / "scores.jsonl", "w") as f:
        for s in SCORE_HISTORY:
            f.write(s.model_dump_json() + "\n")
    logger.info("Saved state: %d alerts, %d scores", len(ALERTS), len(SCORE_HISTORY))


def load_state() -> None:
    """Load ALERTS and SCORE_HISTORY from JSONL files if they exist."""
    state_dir = settings.state_dir
    alerts_path = state_dir / "alerts.jsonl"
    if alerts_path.exists():
        with open(alerts_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ALERTS.append(Alert.model_validate_json(line))
        logger.info("Loaded %d alerts from disk", len(ALERTS))
    scores_path = state_dir / "scores.jsonl"
    if scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    SCORE_HISTORY.append(ScoreEntry.model_validate_json(line))
        logger.info("Loaded %d scores from disk", len(SCORE_HISTORY))
