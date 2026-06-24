"""FastAPI inference service for Cloud Run.

Endpoints
---------
GET  /healthz   -> liveness/readiness (used by Cloud Run + CI smoke test)
GET  /          -> service + model metadata
POST /predict   -> SHD probability + decision for one ECG
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from shd.infer import EnsemblePredictor, SEQ_LEN, N_LEADS, TABULAR_ORDER

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

app = FastAPI(
    title="EchoNext-SHD API",
    version="0.1.0",
    description="Structural heart disease detection from a 12-lead ECG. "
                "Research/demo only — not for clinical use.",
)


@lru_cache(maxsize=1)
def get_predictor() -> EnsemblePredictor:
    # lazy-loaded once per container; keeps cold start off the import path
    return EnsemblePredictor(MODEL_DIR)


class PredictRequest(BaseModel):
    waveform: list[list[float]] = Field(
        ..., description=f"{SEQ_LEN} timesteps x {N_LEADS} leads, z-scored.")
    tabular: list[float] = Field(
        ..., description=f"Raw clinical features in order: {TABULAR_ORDER}")

    @field_validator("waveform")
    @classmethod
    def _check_wave(cls, v):
        if len(v) != SEQ_LEN or any(len(row) != N_LEADS for row in v):
            raise ValueError(f"waveform must be {SEQ_LEN}x{N_LEADS}")
        return v

    @field_validator("tabular")
    @classmethod
    def _check_tab(cls, v):
        if len(v) != len(TABULAR_ORDER):
            raise ValueError(f"tabular must have {len(TABULAR_ORDER)} values")
        return v


class PredictResponse(BaseModel):
    probability: float
    flag: bool
    threshold: float
    risk_band: str


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/")
def root():
    p = get_predictor()
    return {
        "service": "EchoNext-SHD",
        "n_models_in_ensemble": len(p.models),
        "metrics": p.metrics.get("AUROC"),
        "input_contract": {"waveform": f"{SEQ_LEN}x{N_LEADS}", "tabular": TABULAR_ORDER},
        "disclaimer": "Research/demo only. Not a medical device.",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        pred = get_predictor().predict(req.waveform, req.tabular)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictResponse(probability=pred.probability, flag=pred.flag,
                           threshold=pred.threshold, risk_band=pred.risk_band)
