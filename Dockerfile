FROM python:3.11-slim-bookworm

# ffmpeg: decode MP3/M4A/AAC; rubberband-cli: pitch-preserving time-stretch;
# libsndfile1: soundfile; make: `make eval` inside the container;
# fluidsynth + timgm6mb-soundfont: render the public-MIDI Part 3 substitutes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg rubberband-cli libsndfile1 make curl ca-certificates fluidsynth timgm6mb-soundfont \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra dev --no-install-project
RUN playwright install --with-deps chromium

COPY . .
RUN uv sync --frozen --extra dev
# Model weights (Beat This!, Demucs) are fetched on first use into this
# volume-mounted cache, so rebuilding the image does not re-download them.
ENV TORCH_HOME=/cache/torch HF_HOME=/cache/hf XDG_CACHE_HOME=/cache BEAT_AGENT_CACHE=/app/.cache
EXPOSE 8000
CMD ["uvicorn", "beat_agent.app.server:app", "--host", "0.0.0.0", "--port", "8000"]
