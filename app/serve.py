"""Combined Cloud Run entrypoint: FastAPI API + the Gradio demo in one service.

Run with ``uvicorn app.serve:app``. Routes:
    /          interactive Gradio demo
    /predict   prediction API (JSON)
    /healthz   health check
    /info      model metadata
    /docs      OpenAPI / Swagger UI

The pure API lives in ``app.api.main`` and stays importable on its own (that's
what the tests use); this module only adds the UI on top, so deploying both as a
single container needs just one Cloud Run service and one URL.
"""
from __future__ import annotations

import gradio as gr
from app.api.main import app as api_app
from app.demo.app import demo

# explicit API routes (/predict, /healthz, /info, /docs) are registered before
# this mount, so they take precedence; Gradio serves everything else from "/".
app = gr.mount_gradio_app(api_app, demo, path="/")
