"""Vertex AI custom-container prediction server.

Implements the Vertex online-prediction contract on top of EnsemblePredictor:
  GET  $AIP_HEALTH_ROUTE   (default /health)  -> 200 {"status":"ok"}
  POST $AIP_PREDICT_ROUTE  (default /predict) -> {"predictions": [...]}

Request body:
  {"instances": [{"waveform": [[...]x2500], "tabular": [7 raw clinical feats]}, ...]}

Models are baked into the image (MODEL_DIR), so there's no AIP_STORAGE_URI
download to manage. Vertex injects AIP_HTTP_PORT / AIP_*_ROUTE at runtime.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request

from shd.infer import EnsemblePredictor

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

app = FastAPI(title="EchoNext-SHD (Vertex AI)")
_predictor: EnsemblePredictor | None = None


def _get() -> EnsemblePredictor:
    global _predictor
    if _predictor is None:                 # lazy: keep model load off health checks
        _predictor = EnsemblePredictor(MODEL_DIR)
    return _predictor


async def health():
    return {"status": "ok"}


async def predict(request: Request):
    body = await request.json()
    instances = body.get("instances")
    if not isinstance(instances, list):
        raise HTTPException(400, "request body must contain an 'instances' list")
    pred = _get()
    out = []
    for inst in instances:
        try:
            r = pred.predict(inst["waveform"], inst["tabular"])
        except (KeyError, ValueError) as e:
            raise HTTPException(400, f"bad instance: {e}") from e
        out.append({"probability": r.probability, "flag": r.flag,
                    "threshold": r.threshold, "risk_band": r.risk_band})
    return {"predictions": out}


app.add_api_route(HEALTH_ROUTE, health, methods=["GET"])
app.add_api_route(PREDICT_ROUTE, predict, methods=["POST"])
