import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from src.errors.audio import AudioFormatError, AudioLoadError

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg"}
TARGET_SR = 48000


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple resample using linear interpolation (no librosa dependency)."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def load_audio(
    path: str | Path,
    target_sr: int = TARGET_SR,
) -> np.ndarray:
    """Load audio file and resample to target sample rate.

    Returns mono float32 numpy array.
    Uses soundfile directly (avoids librosa fork issues).
    """
    path = Path(path)
    try:
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise AudioFormatError(str(path), f"Supported: {SUPPORTED_FORMATS}")
        audio, orig_sr = sf.read(str(path), dtype="float32")
        # Convert stereo to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Resample if needed
        audio = _resample(audio, orig_sr, target_sr)
        return audio.astype(np.float32)
    except AudioFormatError:
        raise
    except Exception as e:
        raise AudioLoadError(str(path), str(e)) from e


def load_audio_batch(
    paths: list[str | Path],
    target_sr: int = TARGET_SR,
) -> list[np.ndarray]:
    """Load multiple audio files.

    Skips files that fail to load (logs warning).
    """
    results = []
    for p in paths:
        try:
            results.append(load_audio(p, target_sr))
        except (AudioLoadError, AudioFormatError) as e:
            logger.warning("Skipping %s: %s", p, e.message)
    return results
