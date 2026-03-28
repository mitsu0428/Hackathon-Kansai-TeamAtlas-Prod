"""Download AudioSet audio clips from HuggingFace (agkphysics/AudioSet).

No yt-dlp, ffmpeg, or torchcodec required. Reads raw audio bytes from Parquet.

Usage:
    python scripts/download_dataset.py [--limit N] [--source balanced|unbalanced]
"""

import argparse
import io
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

HF_CONFIGS = {
    "balanced": "balanced",
    "unbalanced": "unbalanced",
}

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AudioSet from HuggingFace"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max clips to download (0=all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="balanced",
        choices=["balanced", "unbalanced"],
        help="balanced(~22K clips), unbalanced(~2M clips)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HF_CONFIGS[args.source]
    logger.info(
        "Loading AudioSet '%s' from HuggingFace (streaming, no audio decode)...",
        config,
    )

    # Load with audio decoding DISABLED to avoid torchcodec dependency
    ds = load_dataset(
        "agkphysics/AudioSet",
        config,
        split="train",
        streaming=True,
    )
    # Cast audio column to decode=False so we get raw bytes, not decoded arrays
    ds = ds.cast_column("audio", Audio(decode=False))

    metadata_entries: list[dict] = []
    success = 0
    skipped = 0

    for i, example in enumerate(ds):
        if args.limit and success >= args.limit:
            break

        try:
            video_id = example.get("video_id", f"clip_{i:06d}")
            human_labels = example.get("human_labels", [])
            audio_data = example.get("audio")

            if audio_data is None:
                skipped += 1
                continue

            filename = f"{video_id}.wav"
            output_path = output_dir / filename

            if not output_path.exists():
                # audio_data can be dict with 'bytes', 'path', or 'array'
                if isinstance(audio_data, dict):
                    if "bytes" in audio_data and audio_data["bytes"]:
                        # Raw bytes - decode with soundfile
                        audio_bytes = audio_data["bytes"]
                        data, sr = sf.read(io.BytesIO(audio_bytes))
                        audio_np = np.array(data, dtype=np.float32)
                        sf.write(str(output_path), audio_np, sr)
                    elif "array" in audio_data and audio_data["array"] is not None:
                        # Already decoded array
                        audio_np = np.array(audio_data["array"], dtype=np.float32)
                        sr = audio_data.get("sampling_rate", 16000)
                        sf.write(str(output_path), audio_np, sr)
                    elif "path" in audio_data and audio_data["path"]:
                        # File path - copy
                        import shutil

                        shutil.copy2(audio_data["path"], output_path)
                    else:
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue

            metadata_entries.append({
                "filename": filename,
                "labels": human_labels,
            })
            success += 1

            if success % 100 == 0:
                logger.info("Progress: %d clips saved (%d skipped)", success, skipped)

        except Exception as e:
            logger.warning("Failed to process clip %d: %s", i, e)
            skipped += 1
            continue

    # Save metadata.jsonl
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        "Done: %d clips saved, %d skipped (metadata: %s)",
        success,
        skipped,
        metadata_path,
    )


if __name__ == "__main__":
    main()
