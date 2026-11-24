# syntax=docker/dockerfile:1

# ---- stage 1: build the React dashboard ----
FROM node:22-alpine AS frontend
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- stage 2: python api serving the built dashboard ----
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY --from=frontend /ui/dist ./static

# Pre-download the Whisper weights so the first request isn't slow.
# Containers here have no accelerator, so `auto` resolves to `small`; baking
# in large-v3-turbo would add ~1.6 GB for a model too slow to stream on CPU.
ARG WHISPER_MODEL=small
ARG PREFETCH_MODEL=true
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
      python -c "\
import os; from faster_whisper import WhisperModel; \
name = os.environ['WHISPER_MODEL']; \
name = 'small' if name == 'auto' else name; \
WhisperModel(name, device='cpu', compute_type='int8')"; \
    else \
      echo "skipping model prefetch (PREFETCH_MODEL=$PREFETCH_MODEL)"; \
    fi

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
