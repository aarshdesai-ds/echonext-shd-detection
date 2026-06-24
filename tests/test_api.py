"""API tests. The /healthz test always runs (no model needed); /predict runs
only when artifacts are present."""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
_HAS_MODELS = os.path.isdir(MODEL_DIR) and any(
    f.startswith("ens_seed") for f in (os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else [])
)


def test_healthz():
    from app.api.main import app
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.skipif(not _HAS_MODELS, reason="model artifacts not present")
def test_predict_endpoint():
    from app.api.main import app
    client = TestClient(app)
    body = {
        "waveform": np.random.randn(2500, 12).astype(float).tolist(),
        "tabular": [1, 75, 70, 160, 100, 420, 65],
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    assert 0.0 <= r.json()["probability"] <= 1.0


@pytest.mark.skipif(not _HAS_MODELS, reason="model artifacts not present")
def test_predict_rejects_bad_shape():
    from app.api.main import app
    client = TestClient(app)
    r = client.post("/predict", json={"waveform": [[0.0] * 12], "tabular": [1, 2, 3, 4, 5, 6, 7]})
    assert r.status_code == 422
