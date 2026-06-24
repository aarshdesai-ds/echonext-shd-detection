"""Inference unit tests. Skipped automatically when model artifacts are absent
(e.g. on a fresh clone in CI without the deploy bundle)."""
import os

import numpy as np
import pytest

from shd.infer import EnsemblePredictor, SEQ_LEN, N_LEADS, TABULAR_ORDER

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
_HAS_MODELS = os.path.isdir(MODEL_DIR) and any(
    f.startswith("ens_seed") for f in (os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else [])
)
requires_models = pytest.mark.skipif(not _HAS_MODELS, reason="model artifacts not present")


@pytest.fixture(scope="module")
def predictor():
    return EnsemblePredictor(MODEL_DIR)


def test_tabular_order_is_seven():
    assert len(TABULAR_ORDER) == 7


@requires_models
def test_predict_shape_and_range(predictor):
    wave = np.random.randn(SEQ_LEN, N_LEADS).astype("float32")
    tab = [1, 75, 70, 160, 100, 420, 65]  # raw clinical values
    pred = predictor.predict(wave, tab)
    assert 0.0 <= pred.probability <= 1.0
    assert isinstance(pred.flag, bool)
    assert pred.risk_band in {"low", "moderate", "high"}


@requires_models
def test_predict_accepts_transposed_waveform(predictor):
    wave = np.random.randn(N_LEADS, SEQ_LEN).astype("float32")  # (12, 2500)
    pred = predictor.predict(wave, [0, 80, 80, 150, 95, 410, 55])
    assert 0.0 <= pred.probability <= 1.0


@requires_models
def test_bad_tabular_length_raises(predictor):
    with pytest.raises(ValueError):
        predictor.predict(np.zeros((SEQ_LEN, N_LEADS), "float32"), [1, 2, 3])
