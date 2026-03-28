import logging
from typing import Any

import numpy as np

from src.errors.model import EmbeddingError, ModelLoadError
from src.infra.config import settings

logger = logging.getLogger(__name__)


class ClapModel:
    """Wrapper for CLAP audio embedding model."""

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Load CLAP model and processor."""
        try:
            from transformers import (
                ClapModel as HFClapModel,
            )
            from transformers import (
                ClapProcessor,
            )

            logger.info("Loading CLAP model: %s", settings.CLAP_MODEL_NAME)
            self._processor = ClapProcessor.from_pretrained(settings.CLAP_MODEL_NAME)
            self._model = HFClapModel.from_pretrained(settings.CLAP_MODEL_NAME)
            self._model.to(settings.DEVICE)
            self._model.eval()
            logger.info("CLAP model loaded on %s", settings.DEVICE)
        except Exception as e:
            raise ModelLoadError(settings.CLAP_MODEL_NAME, str(e)) from e

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """Generate embedding for a single audio waveform.

        Returns 1D float32 array.
        """
        try:
            import torch

            inputs = self._processor(
                audio=[audio],
                sampling_rate=sr,
                return_tensors="pt",
            )
            inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.get_audio_features(**inputs)
            # transformers v5+: returns BaseModelOutputWithPooling
            if hasattr(outputs, "pooler_output"):
                tensor = outputs.pooler_output
            else:
                tensor = outputs
            embedding = tensor.cpu().numpy().squeeze(0).astype(np.float32)
            return embedding
        except Exception as e:
            raise EmbeddingError(str(e)) from e

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a text description.

        Returns 1D float32 array (same space as audio embeddings).
        """
        try:
            import torch

            inputs = self._processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.get_text_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                tensor = outputs.pooler_output
            else:
                tensor = outputs
            embedding = tensor.cpu().numpy().squeeze(0).astype(np.float32)
            return embedding
        except Exception as e:
            raise EmbeddingError(str(e)) from e

    def embed_batch(self, audios: list[np.ndarray], sr: int = 48000) -> np.ndarray:
        """Generate embeddings for multiple audio waveforms.

        Returns 2D array (N, dim).
        """
        try:
            import torch

            inputs = self._processor(
                audio=audios,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.get_audio_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                tensor = outputs.pooler_output
            else:
                tensor = outputs
            embeddings = tensor.cpu().numpy().astype(np.float32)
            return embeddings
        except Exception as e:
            raise EmbeddingError(str(e)) from e
