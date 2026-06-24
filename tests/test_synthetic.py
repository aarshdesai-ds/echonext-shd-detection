"""Synthetic ECG generator tests — run in CI without any model artifacts."""
import numpy as np

from shd.synthetic import synthetic_ecg, synthetic_examples


def test_shape_and_finite():
    w = synthetic_ecg(seq_len=2500, n_leads=12, seed=0)
    assert w.shape == (2500, 12)
    assert np.isfinite(w).all()


def test_per_lead_zscored():
    w = synthetic_ecg(seed=1)
    assert np.allclose(w.mean(0), 0, atol=1e-4)
    assert np.allclose(w.std(0), 1, atol=1e-2)


def test_deterministic():
    assert np.array_equal(synthetic_ecg(seed=3), synthetic_ecg(seed=3))


def test_examples_contract():
    ex = synthetic_examples(3)
    assert len(ex) == 3
    for name, wave, tab in ex:
        assert isinstance(name, str)
        assert wave.shape == (2500, 12)
        assert tab.shape == (7,)
