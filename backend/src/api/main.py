import asyncio
import json as _json
import logging
import multiprocessing
from contextlib import asynccontextmanager

# Prevent fork-related crashes with tokenizers/CLAP on macOS
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.deps import (
    ALERTS,
    SCORE_HISTORY,
    SENSORS,
    clap_model,
    faiss_index,
    faiss_indices,
    llm_client,
    load_state,
    save_state,
    state_lock,
)
from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import alerts, baseline, demo, status
from src.infra.config import settings
from src.infra.faiss_index import FaissIndex


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _json.dumps(
            {
                "time": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )


def _setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)


_setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load CLAP model and Faiss index on startup."""
    # Restore persisted state
    try:
        load_state()
    except Exception as e:
        logger.warning("Failed to load persisted state: %s", e)

    # CLAP model
    try:
        clap_model.load()
    except Exception as e:
        logger.warning("CLAP model not loaded: %s", e)

    # Per-sensor Faiss indices
    index_dir = settings.index_dir
    if index_dir.exists():
        for sensor in SENSORS:
            sid = sensor.sensor_id
            idx_path = index_dir / f"{sid}_baseline.faiss"
            meta_path = index_dir / f"{sid}_metadata.jsonl"
            if idx_path.exists():
                try:
                    idx = FaissIndex(512)
                    idx.load(str(idx_path))
                    if meta_path.exists():
                        idx.load_metadata(str(meta_path))
                    faiss_indices[sid] = idx
                    logger.info("Loaded per-sensor index: %s", sid)
                except Exception as e:
                    logger.warning(
                        "Failed to load index for %s: %s",
                        sid,
                        e,
                    )

    # Default/fallback Faiss index
    default_path = index_dir / "default_baseline.faiss"
    if default_path.exists():
        try:
            faiss_index.load(str(default_path))
            # Load metadata
            meta_path = settings.embeddings_dir / "metadata.jsonl"
            faiss_index.load_metadata(str(meta_path))
            # Also register as "default" in indices dict
            faiss_indices.setdefault("default", faiss_index)
        except Exception as e:
            logger.warning("Faiss default index not loaded: %s", e)
    else:
        logger.info(
            "No default index file found at %s, skipping",
            default_path,
        )

    # LLM client
    try:
        llm_client.initialize()
    except Exception as e:
        logger.warning("LLM client not initialized: %s", e)

    # Background detection loop
    from src.use_case.run_detection_loop import (
        init_detection_context,
        run_detection_loop,
    )

    init_detection_context(
        sensors=SENSORS,
        alerts=ALERTS,
        score_history=SCORE_HISTORY,
        clap=clap_model,
        indices=faiss_indices,
        index=faiss_index,
        llm=llm_client,
        lock=state_lock,
    )
    detection_task = asyncio.create_task(run_detection_loop())

    def _on_detection_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Detection loop crashed: %s, restarting...", exc)
            nonlocal detection_task
            detection_task = asyncio.create_task(run_detection_loop())
            detection_task.add_done_callback(_on_detection_done)

    detection_task.add_done_callback(_on_detection_done)

    yield

    detection_task.cancel()
    try:
        await detection_task
    except asyncio.CancelledError:
        pass

    # Persist state on shutdown
    try:
        save_state()
    except Exception as e:
        logger.warning("Failed to save state on shutdown: %s", e)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Spatial Sound Anomaly Detection",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS.split(","),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(status.router)
    app.include_router(alerts.router)
    app.include_router(baseline.router)
    if settings.DEMO_ENABLED:
        app.include_router(demo.router)

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "components": {
                "clap": clap_model.is_loaded,
                "faiss": faiss_index.is_built,
                "llm": llm_client.is_available,
            },
        }

    return app


app = create_app()
