from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {
        "env_file": str(Path(__file__).resolve().parents[3] / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Device
    DEVICE: str = "cpu"
    FAISS_USE_GPU: bool = False

    # Paths
    DATA_DIR: Path = Path(__file__).resolve().parents[3] / "data"
    CLAP_MODEL_NAME: str = "laion/larger_clap_music_and_speech"

    # Anomaly detection
    ANOMALY_THRESHOLD: float = 0.55
    DETECTION_INTERVAL_SEC: int = 300

    # LLM (local Qwen2.5-7B-Instruct)
    LLM_ENABLED: bool = False
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    LLM_MAX_NEW_TOKENS: int = 512

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, v: str) -> str:
        origins = [o.strip() for o in v.split(",")]
        if "*" in origins:
            raise ValueError("Wildcard '*' is not allowed in CORS_ORIGINS")
        return v

    # Demo endpoints
    DEMO_ENABLED: bool = True

    # Debug mode
    DEBUG: bool = False

    @property
    def raw_dir(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def embeddings_dir(self) -> Path:
        return self.DATA_DIR / "embeddings"

    @property
    def index_dir(self) -> Path:
        return self.DATA_DIR / "index"

    @property
    def streams_dir(self) -> Path:
        return self.DATA_DIR / "streams"

    @property
    def state_dir(self) -> Path:
        return self.DATA_DIR / "state"


settings = Settings()
