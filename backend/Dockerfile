FROM python:3.11-slim AS base

# System deps for audio processing (librosa/soundfile)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --create-home app

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (CPU group by default)
RUN uv sync --frozen --extra cpu --no-install-project

# Copy source code
COPY src/ src/

# Own data dir for runtime
RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
