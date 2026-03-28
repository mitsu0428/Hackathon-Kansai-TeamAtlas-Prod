from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from src.domain.anomaly import AnomalyResult

if TYPE_CHECKING:
    from src.domain.ports import EmbeddingPort, IndexPort

logger = logging.getLogger(__name__)


# Silence detection via CLAP text encoder
_silence_vec: np.ndarray | None = None
# True silence → "silence" text sim = 0.55, park nature = 0.22
_SILENCE_THRESHOLD = 0.45


def _is_silence(clap: EmbeddingPort, embedding: np.ndarray) -> bool:
    """Check if audio embedding is similar to 'silence' text."""
    global _silence_vec
    if not hasattr(clap, "embed_text"):
        return False
    try:
        if _silence_vec is None:
            vec = clap.embed_text("silence")
            norm = np.linalg.norm(vec)
            _silence_vec = vec / norm if norm > 0 else vec
        audio_norm = embedding / max(np.linalg.norm(embedding), 1e-8)
        sim = float(np.dot(audio_norm, _silence_vec))
        return sim > _SILENCE_THRESHOLD
    except Exception:
        return False


def detect_anomaly(
    audio: np.ndarray,
    sensor_id: str,
    clap: EmbeddingPort,
    index: IndexPort,
    threshold: float,
    k: int = 5,
) -> AnomalyResult:
    """Embed audio and compare against baseline index."""
    embedding = clap.embed(audio)
    similarities, _, matched_metadata = index.search(embedding, k=k)
    # Use best match (highest similarity) for detection
    best_similarity = float(np.max(similarities[0]))
    baseline_distance = max(0.0, 1.0 - best_similarity)

    # Silence override: if audio sounds like silence, force high distance
    silence_detected = _is_silence(clap, embedding)
    if silence_detected:
        mean_distance = 1.0
    else:
        mean_distance = baseline_distance

    logger.debug(
        "sensor=%s best_sim=%.4f base_dist=%.4f silence=%s final=%.4f",
        sensor_id,
        best_similarity,
        baseline_distance,
        silence_detected,
        mean_distance,
    )

    # Collect labels from matched metadata
    matched_labels: list[str] = []
    for meta in matched_metadata:
        for label in meta.get("labels", []):
            if label not in matched_labels:
                matched_labels.append(label)

    # All baseline categories
    baseline_categories = index.get_all_labels()

    result = AnomalyResult(
        sensor_id=sensor_id,
        timestamp=datetime.now(timezone.utc),
        distance=mean_distance,
        is_anomaly=mean_distance >= threshold,
        threshold=threshold,
        matched_labels=matched_labels,
        baseline_categories=baseline_categories,
    )

    if result.is_anomaly:
        logger.warning(
            "Anomaly detected: sensor=%s distance=%.4f threshold=%.4f matched=%s",
            sensor_id,
            mean_distance,
            threshold,
            matched_labels,
        )
    else:
        logger.info(
            "Normal: sensor=%s distance=%.4f",
            sensor_id,
            mean_distance,
        )

    return result


async def detect_anomaly_async(
    audio: np.ndarray,
    sensor_id: str,
    clap: EmbeddingPort,
    index: IndexPort,
    threshold: float,
    k: int = 5,
) -> AnomalyResult:
    """Async wrapper: run detect_anomaly in executor to avoid blocking."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        partial(
            detect_anomaly,
            audio=audio,
            sensor_id=sensor_id,
            clap=clap,
            index=index,
            threshold=threshold,
            k=k,
        ),
    )
