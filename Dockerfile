# Inference image for Cloud Run (CPU-only — the ensemble is tiny).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
ENV PYTHONPATH=/app/src

# Cloud Run injects $PORT (default 8080). Shell form so $PORT expands.
EXPOSE 8080
CMD exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
