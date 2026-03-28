from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from src.domain.anomaly import AnomalyResult
    from src.domain.intent import Intent
    from src.domain.sensor import Sensor


class EmbeddingPort(Protocol):
    @property
    def is_loaded(self) -> bool: ...
    def embed(
        self,
        audio: np.ndarray,
        sr: int = 48000,
    ) -> np.ndarray: ...
    def embed_batch(
        self,
        audios: list[np.ndarray],
        sr: int = 48000,
    ) -> np.ndarray: ...


class IndexPort(Protocol):
    @property
    def is_built(self) -> bool: ...
    @property
    def ntotal(self) -> int: ...
    def search(
        self,
        vector: np.ndarray,
        k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]: ...
    def build(self, vectors: np.ndarray) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def get_all_labels(self) -> list[str]: ...
    def set_metadata(
        self,
        metadata: list[dict],
    ) -> None: ...


class LLMPort(Protocol):
    @property
    def is_available(self) -> bool: ...
    async def interpret(
        self,
        anomaly: AnomalyResult,
        sensor: Sensor,
        context: dict | None = None,
    ) -> Intent: ...
