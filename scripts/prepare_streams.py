"""Prepare sensor stream directories from downloaded audio data.

Usage:
    # From AudioSet raw data (production) - splits baseline/stream
    python scripts/prepare_streams.py --from-raw \
        --sensors urban,indoor,park --per-sensor 100

    # From ESC-50 data (local dev)
    python scripts/prepare_streams.py --from-esc50
"""
import argparse
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw"
DEFAULT_STREAMS = PROJECT_ROOT / "data" / "streams"

# Add backend to sys.path for shared imports
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from src.domain.categories import LABEL_TO_SENSOR as _LABEL_TO_SENSOR  # noqa: E402

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

# ESC-50 category ID → canonical sensor ID
# ESC-50 category ID → canonical sensor ID (all 50 categories mapped)
_ESC50_TO_SENSOR: dict[str, str] = {
    # urban: vehicles, machines, street noise
    "0": "urban",    # dog_bark (street)
    "3": "urban",    # crying_baby (public space)
    "10": "urban",   # helicopter
    "11": "urban",   # chainsaw
    "12": "urban",   # siren
    "13": "urban",   # car_horn
    "14": "urban",   # engine
    "15": "urban",   # train
    "16": "urban",   # airplane
    "17": "urban",   # fireworks
    "39": "urban",   # glass_breaking
    "40": "urban",   # train
    "49": "urban",   # hand_saw
    # indoor: household, office, human activity
    "4": "indoor",   # clock_tick
    "5": "indoor",   # sneezing
    "7": "indoor",   # clapping
    "8": "indoor",   # breathing
    "9": "indoor",   # coughing
    "18": "indoor",  # hand_saw (workshop)
    "19": "indoor",  # vacuum_cleaner
    "20": "indoor",  # clock_alarm
    "21": "indoor",  # door_knock
    "22": "indoor",  # mouse_click
    "23": "indoor",  # keyboard_typing
    "24": "indoor",  # door_wood_creaks
    "25": "indoor",  # can_opening
    "26": "indoor",  # washing_machine
    "29": "indoor",  # footsteps
    "31": "indoor",  # laughing
    "32": "indoor",  # brushing_teeth
    "33": "indoor",  # snoring
    "34": "indoor",  # drinking_sipping
    "35": "indoor",  # toilet_flush
    "41": "indoor",  # church_bells
    "42": "indoor",  # crackling_fire
    "43": "indoor",  # pouring_water
    # park: nature, animals, weather
    "1": "park",     # rain
    "2": "park",     # sea_waves
    "6": "park",     # insects
    "27": "park",    # cat
    "28": "park",    # water_drops
    "30": "park",    # wind
    "36": "park",    # pig
    "37": "park",    # crow
    "38": "park",    # thunderstorm
    "44": "park",    # rooster
    "45": "park",    # hen
    "46": "park",    # frog
    "47": "park",    # chirping_birds
    "48": "park",    # sheep
}

# ESC-50 category ID → AudioSet-compatible label (must match categories.py LABEL_TO_SENSOR)
_ESC50_TO_LABEL: dict[str, str] = {
    # urban
    "0": "Siren",
    "3": "Siren",
    "10": "Engine",
    "11": "Sawing",
    "12": "Siren",
    "13": "Honking",
    "14": "Engine",
    "15": "Traffic noise, roadway noise",
    "16": "Engine",
    "17": "Siren",
    "39": "Sawing",
    "40": "Traffic noise, roadway noise",
    "49": "Sawing",
    # indoor
    "4": "Air conditioning",
    "5": "Speech",
    "7": "Speech",
    "8": "Speech",
    "9": "Speech",
    "18": "Sawing",
    "19": "Mechanical fan",
    "20": "Air conditioning",
    "21": "Door",
    "22": "Computer keyboard",
    "23": "Computer keyboard",
    "24": "Door",
    "25": "Door",
    "26": "Mechanical fan",
    "29": "Footsteps",
    "31": "Speech",
    "32": "Speech",
    "33": "Speech",
    "34": "Speech",
    "35": "Speech",
    "41": "Door",
    "42": "Air conditioning",
    "43": "Water",
    # park
    "1": "Rain",
    "2": "Water",
    "6": "Insect",
    "27": "Bird",
    "28": "Water",
    "30": "Wind",
    "36": "Bird",
    "37": "Crow",
    "38": "Rain",
    "44": "Bird vocalization, bird call, bird song",
    "45": "Bird vocalization, bird call, bird song",
    "46": "Insect",
    "47": "Chirp, tweet",
    "48": "Bird",
}


def _load_metadata(raw_dir: Path) -> dict[str, list[str]]:
    """Load filename stem -> labels from metadata.jsonl."""
    meta_path = raw_dir / "metadata.jsonl"
    if not meta_path.exists():
        return {}
    mapping: dict[str, list[str]] = {}
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            stem = Path(entry["filename"]).stem
            mapping[stem] = entry.get("labels", [])
    return mapping


def _classify_file(
    stem: str, metadata: dict[str, list[str]]
) -> str | None:
    """Classify a file to a sensor_id based on its labels."""
    labels = metadata.get(stem, [])
    for label in labels:
        sensor = _LABEL_TO_SENSOR.get(label.lower())
        if sensor:
            return sensor
    return None


def prepare_from_raw(
    raw_dir: Path,
    streams_dir: Path,
    sensor_names: list[str],
    per_sensor: int,
    split_ratio: float = 0.7,
    anomaly_ratio: float = 0.0,
) -> None:
    """Split raw audio into baseline (for index) and streams (for testing).

    - split_ratio: fraction of files used as baseline
    - anomaly_ratio: fraction of stream files from different categories
    """
    audio_files = sorted(
        p
        for p in raw_dir.rglob("*")
        if p.suffix.lower() in _AUDIO_EXTS
    )
    if not audio_files:
        logger.warning("No audio files found in %s", raw_dir)
        return

    logger.info(
        "Found %d audio files in %s",
        len(audio_files),
        raw_dir,
    )

    # Load metadata for category-based classification
    metadata = _load_metadata(raw_dir)

    # Classify files by sensor
    by_sensor: dict[str, list[Path]] = defaultdict(list)
    unclassified: list[Path] = []
    for p in audio_files:
        sensor = _classify_file(p.stem, metadata)
        if sensor and sensor in sensor_names:
            by_sensor[sensor].append(p)
        else:
            unclassified.append(p)

    # Distribute unclassified files evenly
    random.shuffle(unclassified)
    for i, p in enumerate(unclassified):
        sensor = sensor_names[i % len(sensor_names)]
        by_sensor[sensor].append(p)

    # Split each sensor's files into baseline and stream
    split_result: dict = {
        "baseline": [],
        "stream": {},
    }

    for sensor_id in sensor_names:
        files = by_sensor.get(sensor_id, [])
        random.shuffle(files)

        n_baseline = max(1, int(len(files) * split_ratio))
        n_baseline = min(n_baseline, per_sensor)
        baseline_files = files[:n_baseline]
        stream_candidates = files[n_baseline:]

        # Record baseline files
        for f in baseline_files:
            split_result["baseline"].append(f.name)

        # Build stream: mostly same-category + some anomaly from other
        sensor_dir = streams_dir / sensor_id
        sensor_dir.mkdir(parents=True, exist_ok=True)

        n_stream = min(per_sensor, len(stream_candidates))
        n_anomaly = max(1, int(n_stream * anomaly_ratio))
        n_normal = n_stream - n_anomaly

        normal_stream = stream_candidates[:n_normal]

        # Anomaly: pick from OTHER sensors' files
        other_files: list[Path] = []
        for other_id in sensor_names:
            if other_id != sensor_id:
                other_files.extend(
                    by_sensor.get(other_id, [])[:20]
                )
        random.shuffle(other_files)
        anomaly_stream = other_files[:n_anomaly]

        stream_files = normal_stream + anomaly_stream
        random.shuffle(stream_files)

        stream_names = []
        for j, src in enumerate(stream_files):
            dst = sensor_dir / f"{j:04d}{src.suffix}"
            shutil.copy2(src, dst)
            stream_names.append(src.name)

        split_result["stream"][sensor_id] = stream_names

        logger.info(
            "Sensor '%s': %d baseline, %d stream "
            "(%d normal + %d anomaly) -> %s",
            sensor_id,
            len(baseline_files),
            len(stream_files),
            len(normal_stream),
            len(anomaly_stream),
            sensor_dir,
        )

    # Save split.json
    split_path = raw_dir / "split.json"
    with open(split_path, "w") as f:
        json.dump(split_result, f, indent=2, ensure_ascii=False)
    logger.info("Split info saved to %s", split_path)


def prepare_from_esc50(
    raw_dir: Path,
    streams_dir: Path,
    max_categories: int = 5,
) -> None:
    """Use ESC-50 category structure as sensor streams.

    ESC-50 filenames: {fold}-{clip_id}-{take}-{category_id}.wav

    Files are grouped into urban/indoor/park sensor directories based on
    ``_ESC50_TO_SENSOR``, and a ``metadata.jsonl`` is generated so that
    downstream tools (run_embed.py, run_update.py) can resolve labels.
    """
    esc50_dir = raw_dir / "esc50"
    if not esc50_dir.exists():
        logger.warning(
            "ESC-50 not found at %s. "
            "Run: python scripts/download_testdata.py",
            esc50_dir,
        )
        return

    audio_files = sorted(esc50_dir.glob("*.wav"))
    if not audio_files:
        logger.warning("No wav files in %s", esc50_dir)
        return

    logger.info("Found %d ESC-50 files", len(audio_files))

    # Group files by sensor using _ESC50_TO_SENSOR
    by_sensor: dict[str, list[Path]] = defaultdict(list)
    metadata_entries: list[dict] = []

    for f in audio_files:
        # Parse filename: {fold}-{clip_id}-{take}-{category_id}.wav
        parts = f.stem.split("-")
        if len(parts) < 4:
            continue
        cat_id = parts[3]
        sensor_id = _ESC50_TO_SENSOR.get(cat_id)
        if sensor_id is None:
            continue
        by_sensor[sensor_id].append(f)
        label = _ESC50_TO_LABEL.get(cat_id, "Unknown")
        metadata_entries.append(
            {"filename": f.name, "labels": [label]}
        )

    # Copy files into sensor directories
    for sensor_id, files in sorted(by_sensor.items()):
        sensor_dir = streams_dir / sensor_id
        sensor_dir.mkdir(parents=True, exist_ok=True)
        for j, src in enumerate(files):
            dst = sensor_dir / f"{j:04d}.wav"
            shutil.copy2(src, dst)
        logger.info(
            "Sensor '%s': %d files -> %s",
            sensor_id,
            len(files),
            sensor_dir,
        )

    # Write metadata.jsonl so run_embed.py / run_update.py can resolve labels
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = raw_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(
        "Wrote %d entries to %s", len(metadata_entries), meta_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare sensor stream directories",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Split raw data into sensors",
    )
    parser.add_argument(
        "--from-esc50",
        action="store_true",
        help="Use ESC-50 categories",
    )
    parser.add_argument(
        "--sensors",
        type=str,
        default="urban,indoor,park",
        help="Comma-separated sensor names (--from-raw)",
    )
    parser.add_argument(
        "--per-sensor",
        type=int,
        default=100,
        help="Files per sensor (--from-raw)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.7,
        help="Fraction of files for baseline (--from-raw)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW),
    )
    parser.add_argument(
        "--streams-dir",
        type=str,
        default=str(DEFAULT_STREAMS),
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    streams_dir = Path(args.streams_dir)

    if not args.from_raw and not args.from_esc50:
        parser.error("Specify --from-raw or --from-esc50")

    if args.from_raw:
        prepare_from_raw(
            raw_dir,
            streams_dir,
            args.sensors.split(","),
            args.per_sensor,
            split_ratio=args.split_ratio,
        )
    elif args.from_esc50:
        prepare_from_esc50(raw_dir, streams_dir)


if __name__ == "__main__":
    main()
