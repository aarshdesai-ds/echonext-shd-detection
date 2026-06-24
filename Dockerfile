# Inference image for Cloud Run (CPU-only — the ensemble is tiny).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models

WORKDIR /app

# requirements-demo.txt = API deps + Gradio/matplotlib for the mounted demo
COPY requirements.txt requirements-demo.txt ./
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
ENV PYTHONPATH=/app/src:/app

# Cloud Run injects $PORT (default 8080). Shell form so $PORT expands.
# app.serve = FastAPI (/predict, /healthz, /info, /docs) + Gradio demo at /.
EXPOSE 8080
CMD exec uvicorn app.serve:app --host 0.0.0.0 --port ${PORT:-8080}
