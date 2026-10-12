FROM python:3.11-slim-bookworm

# ffmpeg: decode MP3/M4A/AAC; rubberband-cli: pitch-preserving time-stretch;
# libsndfile1: soundfile; make: `make eval` inside the container;
# fluidsynth + timgm6mb-soundfont: render the public-MIDI Part 3 substitutes. Both are pinned,
# because the render is an eval input: another FluidSynth version renders the same MIDI
# differently and moves the Part 3 substitute numbers (evals/substitutes/fetch_midi.py).
ARG FLUIDSYNTH_DEB=2.3.1-2
ARG TIMGM6MB_DEB=1.3-5
ARG TIMGM6MB_SHA256=c5378b62028c920cb11e4803327983fee2f2cdff5dc89c708e39da417e51c854
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg rubberband-cli libsndfile1 make curl ca-certificates \
        fluidsynth=${FLUIDSYNTH_DEB} libfluidsynth3=${FLUIDSYNTH_DEB} timgm6mb-soundfont=${TIMGM6MB_DEB} \
    && rm -rf /var/lib/apt/lists/* \
    && echo "${TIMGM6MB_SHA256}  /usr/share/sounds/sf2/TimGM6mb.sf2" | sha256sum -c -

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
