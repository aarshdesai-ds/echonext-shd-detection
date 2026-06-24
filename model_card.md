# Model Card — EchoNext-SHD

## Overview
A 5-seed ensemble of 1D-CNN classifiers that detect **structural heart disease
(SHD)** from a standard 12-lead ECG, fusing the waveform with 7 routine clinical
features (sex, age, ventricular/atrial rate, PR, QRS, QTc). It reimplements and
extends the EchoNext baseline (Nature, 2025).

## Intended use
- **Research, education, and portfolio demonstration only.**
- Decision-support *research* into ECG-based screening for SHD.

## Out of scope / NOT intended for
- **Not a medical device. Not for clinical diagnosis or treatment decisions.**
- Not validated prospectively, across devices, or across populations beyond the
  training distribution.

## Inputs
- Waveform: 2500 timesteps × 12 leads (10 s @ 250 Hz), per-lead z-scored.
- Tabular: 7 raw clinical features, standardized internally by the fitted scaler.

## Performance (held-out test set, n=5442, prevalence 0.43)
| Metric | This model | 95% CI | EchoNext (paper, internal test) |
|---|---|---|---|
| AUROC | 0.842 | [0.832, 0.853] | 0.852 |
| AUPRC | 0.812 | [0.797, 0.827] | 0.785 |
| Brier | 0.158 | — | — |

Operating point (Se@90% specificity, threshold chosen on validation):
sensitivity ≈ 0.57, specificity ≈ 0.91, PPV ≈ 0.82.

Trained on ~72k ECGs — roughly **16× less data** than the 1.2M-pair paper model —
yet statistically on par on AUROC and ahead on AUPRC.

## Training data & limitations
- Derived from the EchoNext modeling tables; train set resampled toward balance
  while validation/test reflect natural ~0.43 prevalence.
- Single source distribution → external generalization is unverified.
- Subgroup performance (age, sex) has not been audited for fairness; do not
  assume equal performance across groups.

## Ethical considerations
SHD screening errors carry real harm (missed disease / overtesting). Any real
deployment would require prospective validation, regulatory review, calibration
monitoring, and clinician oversight. This artifact does none of those.
