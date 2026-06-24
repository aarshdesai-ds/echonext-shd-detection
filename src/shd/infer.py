"""Inference: load the seed ensemble + tabular scaler and predict SHD risk.

Data contract
-------------
* waveform: float array of shape (2500, 12) — 10 s @ 250 Hz, 12 leads, already
  per-lead z-scored (the model was trained on normalized signals). Sample
  inputs in `sample_ecgs.npz` follow this contract.
* tabular: the 7 raw clinical features in this exact order:
      [sex(0/1), ventricular_rate, atrial_rate, pr_interval,
       qrs_duration, qt_corrected, age_at_ecg]
  These are standardized internally with the fitted scaler from training, so
  callers pass raw clinical values.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

SEQ_LEN = 2500
N_LEADS = 12
TABULAR_ORDER = [
    "sex", "ventricular_rate", "atrial_rate", "pr_interval",
    "qrs_duration", "qt_corrected", "age_at_ecg",
]
# indices of the 6 continuous features that get standardized (sex is passthrough)
_CONT_IDX = [1, 2, 3, 4, 5, 6]


@dataclass
class Prediction:
    probability: float          # calibrated SHD probability in [0, 1]
    flag: bool                  # decision at the Se@90%-specificity operating point
    threshold: float
    risk_band: str              # "low" | "moderate" | "high" (display only)


class EnsemblePredictor:
    """Loads N seed models + scaler + operating threshold; averages with TTA."""

    def __init__(self, model_dir: str, tta_shifts=(0, -75, 75)):
        # heavy imports are local so the module imports cheaply (e.g. for tests)
        import keras

        from .model import CUSTOM_OBJECTS

        self.model_dir = model_dir
        self.tta_shifts = tta_shifts

        paths = sorted(glob.glob(os.path.join(model_dir, "ens_seed*.keras")))
        if not paths:
            raise FileNotFoundError(
                f"No ens_seed*.keras checkpoints in {model_dir!r}. "
                "Download the deploy bundle from training (see infra/DEPLOY.md)."
            )
        self.models = [keras.models.load_model(p, custom_objects=CUSTOM_OBJECTS)
                       for p in paths]

        scaler = np.load(os.path.join(model_dir, "tabular_scaler.npz"))
        self._mean = scaler["mean"].astype(np.float32)   # shape (6,)
        self._std = scaler["std"].astype(np.float32)      # shape (6,)

        with open(os.path.join(model_dir, "metrics.json")) as f:
            self.metrics = json.load(f)
        self.threshold = float(self.metrics["operating_point"]["threshold"])

    # ── preprocessing ─────────────────────────────────────────────────────────
    def _prep_tabular(self, tabular) -> np.ndarray:
        t = np.asarray(tabular, dtype=np.float32).reshape(-1)
        if t.shape[0] != len(TABULAR_ORDER):
            raise ValueError(f"tabular must have {len(TABULAR_ORDER)} values "
                             f"({TABULAR_ORDER}); got {t.shape[0]}")
        out = t.copy()
        out[_CONT_IDX] = (t[_CONT_IDX] - self._mean) / self._std
        return out.reshape(1, -1)

    def _prep_waveform(self, waveform) -> np.ndarray:
        w = np.asarray(waveform, dtype=np.float32)
        if w.shape == (N_LEADS, SEQ_LEN):       # accept (12, 2500) too
            w = w.T
        if w.shape != (SEQ_LEN, N_LEADS):
            raise ValueError(f"waveform must be ({SEQ_LEN},{N_LEADS}); got {w.shape}")
        return w[None, ...]

    # ── prediction ────────────────────────────────────────────────────────────
    def predict(self, waveform, tabular) -> Prediction:
        x = self._prep_waveform(waveform)
        t = self._prep_tabular(tabular)

        per_model = []
        for m in self.models:
            shifted = [m.predict([np.roll(x, s, axis=1) if s else x, t], verbose=0).ravel()[0]
                       for s in self.tta_shifts]
            per_model.append(float(np.mean(shifted)))
        prob = float(np.mean(per_model))

        flag = prob >= self.threshold
        band = "high" if prob >= 0.66 else "moderate" if prob >= 0.33 else "low"
        return Prediction(probability=prob, flag=bool(flag),
                          threshold=self.threshold, risk_band=band)
