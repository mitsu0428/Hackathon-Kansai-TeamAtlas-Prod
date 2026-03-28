"""Run CLAP embedding on audio files.

Usage:
    python batch/jobs/run_embed.py [--limit N] [--batch-size 32]
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Keep in sync with src.infra.audio_loader.SUPPORTED_FORMATS
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_raw_metadata(raw_dir: Path) -> dict[str, list[str]]:
    """Load filename -> labels mapping from metadata.jsonl."""
    metadata_path = raw_dir / "metadata.jsonl"
    if not metadata_path.exists():
        logger.info("No metadata.jsonl found, labels will be empty")
        return {}
    mapping: dict[str, list[str]] = {}
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # stem (without extension) -> labels
            stem = Path(entry["filename"]).stem
            mapping[stem] = entry.get("labels", [])
    logger.info("Loaded metadata for %d files", len(mapping))
    return mapping


def _load_split(raw_dir: Path) -> set[str] | None:
    """Load split.json and return set of baseline filenames (stems).

    Returns None if split.json does not exist (all files are used).
    """
    split_path = raw_dir / "split.json"
    if not split_path.exists():
        return None
    with open(split_path) as f:
        split = json.load(f)
    baseline_files = set()
    for entry in split.get("baseline", []):
        baseline_files.add(Path(entry).stem)
    logger.info(
        "Split loaded: %d baseline files", len(baseline_files)
    )
    return baseline_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CLAP embeddings"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files to process (0=all)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    # Add backend to sys.path so imports work
    sys.path.insert(0, str(project_root / "backend"))

    from src.infra.audio_loader import load_audio
    from src.infra.clap_model import ClapModel
    from src.infra.config import settings

    input_dir = (
        Path(args.input_dir) if args.input_dir else settings.raw_dir
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else settings.embeddings_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata and split info
    raw_metadata = _load_raw_metadata(input_dir)
    baseline_stems = _load_split(input_dir)

    # Collect audio files
    audio_files = sorted(
        p
        for p in input_dir.rglob("*")
        if p.suffix.lower() in _AUDIO_EXTS
    )

    # Filter to baseline-only if split exists
    if baseline_stems is not None:
        before = len(audio_files)
        audio_files = [
            p for p in audio_files
            if p.stem in baseline_stems
        ]
        logger.info(
            "Filtered to baseline: %d -> %d files",
            before,
            len(audio_files),
        )

    if args.limit:
        audio_files = audio_files[: args.limit]

    if not audio_files:
        logger.warning("No audio files found in %s", input_dir)
        return

    logger.info(
        "Found %d audio files in %s", len(audio_files), input_dir
    )

    # Load model
    clap = ClapModel()
    clap.load()

    # Process in batches
    all_embeddings = []
    all_filenames = []
    all_metadata = []
    batch_size = args.batch_size

    for i in range(0, len(audio_files), batch_size):
        batch_paths = audio_files[i : i + batch_size]
        batch_audios = []
        batch_names = []
        batch_meta = []

        for p in batch_paths:
            try:
                audio = load_audio(p)
                batch_audios.append(audio)
                batch_names.append(p.stem)
                batch_meta.append({
                    "filename": p.name,
                    "labels": raw_metadata.get(p.stem, []),
                })
            except Exception as e:
                logger.warning("Skipping %s: %s", p.name, e)

        if not batch_audios:
            continue

        embeddings = clap.embed_batch(batch_audios)
        all_embeddings.append(embeddings)
        all_filenames.extend(batch_names)
        all_metadata.extend(batch_meta)

        processed = min(i + batch_size, len(audio_files))
        logger.info(
            "Progress: %d/%d", processed, len(audio_files)
        )

    if not all_embeddings:
        logger.warning("No embeddings generated")
        return

    # Save embeddings (atomic write via tmp + os.replace)
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    output_path = output_dir / "embeddings.npy"
    # np.save auto-appends .npy, so use stem without extension for tmp
    tmp_path = output_dir / "embeddings.tmp"
    np.save(str(tmp_path), embeddings_array)
    # np.save creates embeddings.tmp.npy
    os.replace(str(tmp_path) + ".npy", output_path)

    names_path = output_dir / "filenames.txt"
    tmp_names_path = Path(str(names_path) + ".tmp")
    tmp_names_path.write_text("\n".join(all_filenames))
    os.replace(tmp_names_path, names_path)

    # Save metadata.jsonl (index-aligned with embeddings)
    metadata_path = output_dir / "metadata.jsonl"
    tmp_metadata_path = Path(str(metadata_path) + ".tmp")
    with open(tmp_metadata_path, "w") as f:
        for idx, meta in enumerate(all_metadata):
            entry = {"index": idx, **meta}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp_metadata_path, metadata_path)

    logger.info(
        "Saved %d embeddings (%s) to %s",
        len(all_filenames),
        embeddings_array.shape,
        output_path,
    )
    logger.info(
        "Saved metadata for %d entries to %s",
        len(all_metadata),
        metadata_path,
    )


if __name__ == "__main__":
    main()
