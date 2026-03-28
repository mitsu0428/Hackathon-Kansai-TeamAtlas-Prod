"""Update baseline index with new embeddings.

Usage:
    python batch/jobs/run_update.py [--sensor-id default]
    python batch/jobs/run_update.py --all-sensors
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Add backend to sys.path early so shared imports work
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))


def _load_metadata(embeddings_dir: Path) -> list[dict]:
    """Load embedding-aligned metadata."""
    meta_path = embeddings_dir / "metadata.jsonl"
    if not meta_path.exists():
        return []
    entries = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _classify_entry(
    entry: dict, label_to_sensor: dict[str, str]
) -> str | None:
    """Classify a metadata entry to sensor_id."""
    for label in entry.get("labels", []):
        sensor = label_to_sensor.get(label.lower())
        if sensor:
            return sensor
    return None


def _build_single_index(
    vectors: np.ndarray,
    metadata: list[dict],
    sensor_id: str,
    index_dir: Path,
) -> None:
    """Build and save a single Faiss index with metadata."""
    from src.infra.faiss_index import FaissIndex

    if vectors.shape[0] == 0:
        logger.warning(
            "No vectors for sensor %s, skipping", sensor_id
        )
        return

    dimension = vectors.shape[1]
    index = FaissIndex(dimension)
    index.build(vectors)
    index.set_metadata(metadata)

    index_path = index_dir / f"{sensor_id}_baseline.faiss"
    index.save(str(index_path))

    meta_path = index_dir / f"{sensor_id}_metadata.jsonl"
    index.save_metadata(str(meta_path))

    logger.info(
        "Sensor '%s': index with %d vectors saved to %s",
        sensor_id,
        vectors.shape[0],
        index_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update baseline index"
    )
    parser.add_argument(
        "--sensor-id", type=str, default="default"
    )
    parser.add_argument(
        "--all-sensors",
        action="store_true",
        help="Build per-sensor indices based on category mapping",
    )
    args = parser.parse_args()

    from src.domain.categories import (
        CATEGORY_SENSOR_MAP,
        LABEL_TO_SENSOR,
    )
    from src.infra.config import settings

    embeddings_dir = settings.embeddings_dir
    index_dir = settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    # Load existing embeddings
    main_path = embeddings_dir / "embeddings.npy"
    if not main_path.exists():
        logger.error("No embeddings found at %s", main_path)
        sys.exit(1)

    vectors = np.load(str(main_path))
    logger.info(
        "Loaded existing embeddings: shape=%s", vectors.shape
    )

    # Check for new embeddings to merge
    new_path = embeddings_dir / "new" / "embeddings.npy"
    if new_path.exists():
        new_vectors = np.load(str(new_path))
        logger.info(
            "Found new embeddings: shape=%s", new_vectors.shape
        )
        vectors = np.concatenate(
            [vectors, new_vectors], axis=0
        )
        np.save(str(main_path), vectors)
        new_path.unlink()
        logger.info(
            "Merged embeddings: shape=%s", vectors.shape
        )

    # Load metadata
    metadata = _load_metadata(embeddings_dir)

    if args.all_sensors and metadata:
        # Build per-sensor indices
        sensor_vectors: dict[str, list[int]] = {
            sid: [] for sid in CATEGORY_SENSOR_MAP
        }

        for i, entry in enumerate(metadata):
            if i >= vectors.shape[0]:
                break
            sensor = _classify_entry(entry, LABEL_TO_SENSOR)
            if sensor and sensor in sensor_vectors:
                sensor_vectors[sensor].append(i)

        for sensor_id, indices in sensor_vectors.items():
            if not indices:
                logger.warning(
                    "No files classified for sensor %s",
                    sensor_id,
                )
                continue
            idx_array = np.array(indices)
            sensor_vecs = vectors[idx_array]
            sensor_meta = [metadata[i] for i in indices]
            _build_single_index(
                sensor_vecs, sensor_meta, sensor_id, index_dir
            )

        # Also build "default" index with ALL vectors
        _build_single_index(
            vectors, metadata, "default", index_dir
        )

        logger.info(
            "Built %d per-sensor indices + default",
            sum(1 for v in sensor_vectors.values() if v),
        )
    else:
        # Single index mode (original behavior)
        _build_single_index(
            vectors, metadata, args.sensor_id, index_dir
        )


if __name__ == "__main__":
    main()
