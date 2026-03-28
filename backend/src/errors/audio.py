from src.errors.base import AppError


class AudioLoadError(AppError):
    def __init__(self, path: str, detail: str = "") -> None:
        super().__init__(
            f"Failed to load audio: {path}. {detail}",
            status_code=422,
        )


class AudioFormatError(AppError):
    def __init__(self, path: str, detail: str = "") -> None:
        super().__init__(
            f"Unsupported audio format: {path}. {detail}",
            status_code=422,
        )
