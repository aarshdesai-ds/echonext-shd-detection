"""Procedurally-generated synthetic 12-lead ECGs for the PUBLIC demo.

These signals are synthesized from simple PQRST morphology — they are **not**
records from the PhysioNet EchoNext dataset, so the public demo redistributes no
restricted health data. Predictions on synthetic (or arbitrary uploaded) inputs
are **illustrative only** and not clinically meaningful; validated metrics are
reported on the held-out dataset test set in the repo README.
"""
from __future__ import annotations

import numpy as np

FS = 250  # Hz


def _gauss(t: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def _beat(rr_samples: int) -> np.ndarray:
    """One synthetic PQRST beat over `rr_samples` points (RR interval)."""
    t = np.linspace(0.0, 1.0, max(rr_samples, 50), endpoint=False)
    return (_gauss(t, 0.15, 0.020, 0.10)    # P
            + _gauss(t, 0.33, 0.006, -0.08)  # Q
            + _gauss(t, 0.35, 0.008, 1.00)   # R
            + _gauss(t, 0.37, 0.008, -0.25)  # S
            + _gauss(t, 0.60, 0.040, 0.25))  # T


def synthetic_ecg(seq_len: int = 2500, n_leads: int = 12,
                  heart_rate: float = 72.0, seed: int = 0) -> np.ndarray:
    """Synthetic, per-lead z-scored ECG of shape (seq_len, n_leads). Not real data."""
    rng = np.random.default_rng(seed)
    rr = int(FS * 60.0 / heart_rate)

    sig = np.zeros(seq_len)
    i = 0
    while i < seq_len:
        b = _beat(rr + int(rng.normal(0, rr * 0.03)))   # small RR jitter
        end = min(i + len(b), seq_len)
        sig[i:end] += b[:end - i]
        i += len(b)

    # per-lead gain/polarity (aVR ~ inverted) + baseline wander + noise, then z-score
    gain = np.array([1.0, 1.2, 0.8, -0.7, 0.5, 0.9, 0.6, 1.1, 1.3, 1.2, 1.0, 0.8])
    base_t = np.linspace(0.0, 1.0, seq_len)
    out = np.empty((seq_len, n_leads), dtype=np.float32)
    for lead in range(n_leads):
        wander = 0.05 * np.sin(2 * np.pi * rng.uniform(0.2, 0.5) * base_t * 10)
        noise = rng.normal(0, 0.02, seq_len)
        x = gain[lead % len(gain)] * sig + wander + noise
        out[:, lead] = (x - x.mean()) / (x.std() + 1e-6)
    return out


def synthetic_examples(k: int = 3):
    """A few deterministic synthetic ECGs + placeholder clinical features.

    Returns list of (name, waveform (2500,12), tabular (7,)). Tabular order matches
    shd.infer.TABULAR_ORDER: [sex, ventricular_rate, atrial_rate, pr_interval,
    qrs_duration, qt_corrected, age_at_ecg].
    """
    hrs = [62, 78, 95]
    out = []
    for j in range(k):
        hr = hrs[j % len(hrs)]
        wave = synthetic_ecg(heart_rate=hr, seed=j)
        tab = np.array([j % 2, hr, hr, 160.0, 95.0, 420.0, 55 + 10 * j], dtype=np.float32)
        out.append((f"Synthetic example {j + 1} (HR~{hr})", wave, tab))
    return out
