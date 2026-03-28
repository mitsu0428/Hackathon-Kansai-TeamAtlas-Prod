from src.errors.base import AppError


class IndexBuildError(AppError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            f"Failed to build Faiss index. {detail}",
            status_code=500,
        )


class IndexSearchError(AppError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            f"Faiss search failed. {detail}",
            status_code=500,
        )


class IndexIOError(AppError):
    def __init__(self, path: str, detail: str = "") -> None:
        super().__init__(
            f"Faiss index I/O error: {path}. {detail}",
            status_code=500,
        )
