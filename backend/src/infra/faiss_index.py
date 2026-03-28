from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.errors.index import (
    IndexBuildError,
    IndexIOError,
    IndexSearchError,
)
from src.infra.config import settings

logger = logging.getLogger(__name__)

NLIST = 256


def _import_faiss() -> Any:
    """Lazy import faiss to avoid early fork issues with tokenizers."""
    import faiss

    return faiss


class FaissIndex:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index: Any | None = None
        self._metadata: list[dict] | None = None
        self._lock = threading.Lock()

    @property
    def is_built(self) -> bool:
        return self._index is not None

    @property
    def ntotal(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    def build(self, vectors: np.ndarray) -> None:
        try:
            faiss = _import_faiss()
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            # L2正規化してコサイン距離ベースにする
            faiss.normalize_L2(vectors)
            n = vectors.shape[0]
            nlist = min(NLIST, max(1, n // 39))
            logger.info(
                "Building IVF index: %d vectors, dim=%d, nlist=%d",
                n,
                self._dimension,
                nlist,
            )
            quantizer = faiss.IndexFlatIP(self._dimension)
            metric = faiss.METRIC_INNER_PRODUCT
            index = faiss.IndexIVFFlat(quantizer, self._dimension, nlist, metric)
            index.train(vectors)
            index.add(vectors)
            if settings.FAISS_USE_GPU:
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                except AttributeError:
                    logger.warning("faiss-gpu not available, falling back to CPU")
            self._index = index
            logger.info("Index built: %d vectors", self._index.ntotal)
        except Exception as e:
            raise IndexBuildError(str(e)) from e

    def search(self, vector: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        if self._index is None:
            raise IndexSearchError("Index not built")
        try:
            vector = np.ascontiguousarray(vector, dtype=np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            # L2正規化（コサイン類似度ベースのインデックスに合わせる）
            faiss = _import_faiss()
            faiss.normalize_L2(vector)
            nprobe = min(64, getattr(self._index, "nlist", 64))
            with self._lock:
                if hasattr(self._index, "setNumProbes"):
                    self._index.setNumProbes(nprobe)
                else:
                    self._index.nprobe = nprobe
                distances, indices = self._index.search(vector, k)

            # メタデータの解決
            matched_metadata: list[dict] = []
            if self._metadata is not None:
                for idx in indices[0]:
                    if 0 <= idx < len(self._metadata):
                        matched_metadata.append(self._metadata[idx])
                    else:
                        matched_metadata.append({})
            else:
                matched_metadata = [{} for _ in indices[0]]

            return distances, indices, matched_metadata
        except IndexSearchError:
            raise
        except Exception as e:
            raise IndexSearchError(str(e)) from e

    def save(self, path: str) -> None:
        if self._index is None:
            raise IndexIOError(path, "No index to save")
        try:
            faiss = _import_faiss()
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            index_to_save = self._index
            if settings.FAISS_USE_GPU:
                try:
                    index_to_save = faiss.index_gpu_to_cpu(self._index)
                except AttributeError:
                    pass  # already on CPU
            faiss.write_index(index_to_save, str(path))
            logger.info("Index saved to %s", path)
        except IndexIOError:
            raise
        except Exception as e:
            raise IndexIOError(path, str(e)) from e

    def load(self, path: str) -> None:
        try:
            faiss = _import_faiss()
            if not Path(path).exists():
                raise IndexIOError(path, "File not found")
            index = faiss.read_index(str(path))
            if settings.FAISS_USE_GPU:
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                except AttributeError:
                    logger.warning("faiss-gpu not available, falling back to CPU")
            self._index = index
            self._dimension = index.d
            logger.info(
                "Index loaded from %s (%d vectors)",
                path,
                self._index.ntotal,
            )
        except IndexIOError:
            raise
        except Exception as e:
            raise IndexIOError(path, str(e)) from e

    def load_metadata(self, path: str) -> None:
        """Load index-aligned metadata from JSONL file."""
        p = Path(path)
        if not p.exists():
            logger.info("No metadata file at %s", path)
            return
        entries: list[dict] = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        self._metadata = entries
        logger.info(
            "Loaded metadata: %d entries from %s",
            len(entries),
            path,
        )

    def save_metadata(self, path: str) -> None:
        """Save metadata to JSONL file."""
        if self._metadata is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self._metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Metadata saved to %s", path)

    def set_metadata(self, metadata: list[dict]) -> None:
        """Set metadata directly (for programmatic use)."""
        self._metadata = metadata

    def get_all_labels(self) -> list[str]:
        """Get all unique labels from metadata."""
        if self._metadata is None:
            return []
        labels: set[str] = set()
        for entry in self._metadata:
            for label in entry.get("labels", []):
                labels.add(label)
        return sorted(labels)
