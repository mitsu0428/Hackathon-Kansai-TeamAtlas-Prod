"""Download ESC-50 dataset for local development/testing.

ESC-50 is a small (~600MB) environmental sound classification dataset.
Usage: python scripts/download_testdata.py [--output-dir DIR]
"""
import argparse
import logging
import subprocess
import tarfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ESC50_URL = (
    "https://github.com/karolpiczak/ESC-50/"
    "archive/refs/heads/master.tar.gz"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ESC-50 test dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir.parent / "esc50.tar.gz"

    if not archive_path.exists():
        logger.info("Downloading ESC-50 (~600MB)...")
        subprocess.run(
            ["curl", "-fSL", "-o", str(archive_path), ESC50_URL],
            check=True,
        )
        logger.info("Download complete: %s", archive_path)
    else:
        logger.info("Archive already exists: %s", archive_path)

    # Extract audio files
    audio_dir = output_dir / "esc50"
    if not audio_dir.exists():
        logger.info("Extracting audio files...")
        with tarfile.open(archive_path) as tar:
            # Only extract .wav files from audio/ directory
            members = [
                m
                for m in tar.getmembers()
                if m.name.endswith(".wav") and "/audio/" in m.name
            ]
            for member in members:
                if member.issym() or member.islnk():
                    continue
                member.name = Path(member.name).name
                tar.extract(member, audio_dir)  # noqa: S202
            logger.info(
                "Extracted %d audio files to %s",
                len(members),
                audio_dir,
            )
    else:
        logger.info(
            "Audio directory already exists: %s", audio_dir
        )

    logger.info("Done. ESC-50 data available at %s", audio_dir)


if __name__ == "__main__":
    main()
