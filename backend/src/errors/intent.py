from src.errors.base import AppError


class IntentGenerationError(AppError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            f"Intent generation failed. {detail}",
            status_code=500,
        )


class IntentParseError(AppError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            f"Failed to parse LLM response. {detail}",
            status_code=500,
        )
