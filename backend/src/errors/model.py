from src.errors.base import AppError


class ModelLoadError(AppError):
    def __init__(self, model_name: str, detail: str = "") -> None:
        super().__init__(
            f"Failed to load model: {model_name}. {detail}",
            status_code=500,
        )


class EmbeddingError(AppError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            f"Embedding generation failed. {detail}",
            status_code=500,
        )
