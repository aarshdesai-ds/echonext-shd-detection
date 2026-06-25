# EchoNext-SHD — Structural Heart Disease Detection from 12-Lead ECGs

An end-to-end deep-learning study that detects **echocardiogram-confirmed structural heart disease (SHD)** from a standard 12-lead ECG, benchmarked against the published EchoNext model — covering architecture design, rigorous evaluation, an honest model ablation, and a production-style serving stack.

> ⚠️ **Research and educational use only. Not a medical device and not for clinical decision-making.** See [model_card.md](model_card.md).

## Table of Contents
- [Overview](#overview)
- [Research Questions](#research-questions)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Ideas for Extension](#ideas-for-extension)

## Overview

This project reimplements and improves a model that screens for **structural heart disease** — conditions normally confirmed by echocardiography (reduced ejection fraction, valvular disease, wall-thickening, etc.) — directly from a cheap, ubiquitous 12-lead ECG. It is built on the public **PhysioNet EchoNext** benchmark and compared against the model described in the EchoNext paper (Nature, 2025).

Three pillars:

- **A corrected, modern architecture** — a 1D-CNN residual "tokenizer" with attention pooling, replacing a broken baseline whose receptive field was too small to see a full heartbeat.
- **Multimodal fusion + ensembling** — the ECG waveform is fused with 7 routine clinical features (age, sex, ECG intervals), and 5 seeds are averaged with test-time augmentation.
- **Rigorous evaluation + delivery** — bootstrap confidence intervals, calibration, a clinically meaningful operating point, a controlled ablation, and a containerized FastAPI + Gradio service with CI and a versioned model registry.

The core question: **how close can a compact, reproducible model get to the published EchoNext metrics on the public benchmark — and what actually drives the performance?**

## Research Questions

1. Can a compact CNN **match the published EchoNext metrics** on the public PhysioNet benchmark, despite far less training data?
2. What actually limited the original baseline — **architecture or data**?
3. Does an **explicit temporal module** (Transformer / GRU) improve detection at this data scale, or is a convolutional receptive field enough?
4. Does **fusing clinical features** with the ECG waveform improve detection over the waveform alone?
5. Is the model **well-calibrated**, and usable at a clinically meaningful operating point (high sensitivity at fixed specificity)?
6. Can the whole pipeline be packaged as a **reproducible, deployable** service while respecting the dataset's restricted-data license?

## Dataset Description

**EchoNext (PhysioNet)** — a de-identified collection of 12-lead ECGs paired with echocardiogram-confirmed SHD labels, from Columbia University Irving Medical Center. Accessed under the PhysioNet **Restricted Health Data License** (credentialed; **not redistributed** in this repo).

**ECG waveforms** (`EchoNext_{train,val,test}_waveforms.npy`)

| Property | Value |
|---|---|
| Shape | `(N, 1, 2500, 12)` |
| Sampling | 250 Hz, 10 s, 12 leads |
| Preprocessing | per-lead z-scored, clipped at ±10 |

**Clinical / tabular features** (`EchoNext_{split}_tabular_features.npy`, 7 columns)

| Feature | Description |
|---|---|
| `sex` | 0 / 1 (binary, passthrough) |
| `ventricular_rate` | bpm (standardized) |
| `atrial_rate` | bpm (standardized) |
| `pr_interval` | ms (standardized) |
| `qrs_duration` | ms (standardized) |
| `qt_corrected` | QTc, ms (standardized) |
| `age_at_ecg` | years (standardized) |

**Labels** — `shd_label` (binary; SHD present vs. absent).

**Splits**

| Split | ECGs | Prevalence (positive) |
|---|---|---|
| Train | 72,475 | 0.52 (resampled toward balance) |
| Validation | 4,626 | 0.43 (natural) |
| Test | 5,442 | 0.43 (natural) |

## Methodology

### Baseline diagnosis
The original 2D-CNN used `(1,1)` convolution kernels after its first layer, giving an effective temporal receptive field of **~48 ms** — too small to see a complete QRS/ST/T complex. It plateaued at ~0.81 validation AUROC. The fix was architectural, not more data.

### Preprocessing
Waveforms arrive pre–z-scored, reshaped to `(2500, 12)` (channels-last for `Conv1D`). The 6 continuous tabular features are standardized with a scaler fit on the training set; `sex` is passed through. At inference the service accepts **raw** clinical values and standardizes internally.

### Model architecture (per seed)
```
waveform (2500×12) ──► Conv1D stem (k=17, stride 2)
                  ──► 4 residual stages, k=16, channels [64,128,192,256], stride-2 downsampling
                       → ~79 "beat-level" tokens × 256 channels
                  ──► learned-query attention pooling → 256-d ECG embedding
clinical (7) ─────► MLP [32 → 16]
                  ──► concat(ECG, clinical) → Dense(64) → Dense(1, float32) → sigmoid
```
~5M parameters; mixed-precision (`float16`) compute with a `float32` output head.

### Training recipe
- Loss: binary cross-entropy (optional label smoothing / focal); model selection on **validation AUPRC**.
- Optimizer: AdamW, gradient clipping, **3-epoch warmup → cosine decay**, peak LR `3e-4`.
- **ECG augmentation** (train only): amplitude scaling, Gaussian noise, baseline wander, random lead dropout, time shift.
- Output bias initialized to the training log-odds for fast, calibrated convergence.

### Ensemble + test-time augmentation
Final prediction averages 5 seeds, each averaged over small time-shifts:
```
p = mean_over_seeds( mean_over_shifts s∈{0,−75,+75}( model( roll(x, s) ) ) )
```

### Ablation
The temporal module is swappable (`transformer` / `gru` / `none`) with everything else held fixed, to isolate its contribution.

### Evaluation
- **AUROC** and **AUPRC** with **bootstrap 95% CIs** (2,000 resamples).
- **Calibration** via Brier score and reliability curve.
- **Operating point**: sensitivity at fixed 90% specificity, with the threshold chosen on **validation** and applied to test (no leakage).

## Key Findings

### 1. The bottleneck was receptive field, not data
Replacing the `(1,1)` kernels with a deep `Conv1D` residual stack (large receptive field + attention pooling) moved validation AUROC from ~0.81 to ~0.83 — and AUPRC up to parity with the published model — before any ensembling.

### 2. Ensembling + TTA delivered the final gain
A 5-seed ensemble with test-time augmentation reached **test AUROC 0.842 / AUPRC 0.812**, above any single seed (0.832–0.839) and with improved calibration (Brier 0.176 → 0.158).

### 3. Explicit temporal attention was a statistical wash
In a controlled ablation, Transformer and GRU temporal heads were **indistinguishable** from a pure CNN (all ~0.83 AUROC, within the bootstrap CI). Convolutional receptive field + attention pooling already capture the temporal structure at this data scale; attention is data-hungry and didn't pay off. *Reported, not hidden.*

### 4. The model is reasonably calibrated
Brier **0.158** (vs. ~0.245 for a no-skill model at this prevalence). At the Se@90%-specificity operating point, the screen reaches **PPV ≈ 0.82** — credible for a triage tool.

### 5. Comparable to the published model — honestly framed
| | This model (public test) | EchoNext paper |
|---|---|---|
| AUROC | 0.842 (95% CI 0.832–0.853) | 0.852 |
| AUPRC | 0.812 (95% CI 0.797–0.827) | 0.785 |

The AUROC CI **includes** the paper's 0.852 and the AUPRC point estimate **exceeds** 0.785 — using ~16× less training data. This is **not head-to-head**: the paper trained on 1.2M ECGs from a private multi-site cohort and tested on its own internal set, so the takeaway is *comparability on the public benchmark*, not a superiority claim.

### 6. Multimodal fusion helps and is nearly free
Fusing the 7 clinical features with the ECG embedding improves discrimination and calibration over the waveform alone — the same setup the original work uses.

## Project Structure
```
echonext-shd/
│
├── notebooks/                  # training + evaluation (reproduces the model bundle)
│
├── src/shd/
│   ├── model.py                # architecture + custom Keras layers
│   ├── infer.py                # EnsemblePredictor (5 models + TTA + scaler)
│   ├── synthetic.py            # DUA-safe synthetic ECGs for the public demo
│   └── registry.py             # versioned model store (GCS / HF Hub)
│
├── app/
│   ├── api/main.py             # FastAPI service (/predict, /healthz, /info, /docs)
│   ├── demo/app.py             # Gradio demo
│   └── serve.py                # combined entrypoint: API + demo mounted at /
│
├── tests/                      # pytest (CI gate; model-dependent tests self-skip)
├── infra/                      # deploy runbooks (Cloud Run, EC2) + bootstrap script
├── .github/workflows/          # ci.yml (ruff + pytest), deploy.yml
├── Dockerfile · model_card.md · pyproject.toml · Makefile
```

## Requirements
- Python 3.11
- tensorflow / keras (training: GPU; serving: `tensorflow-cpu`)
- numpy, scikit-learn
- fastapi, uvicorn, pydantic
- gradio, matplotlib (demo)

Install (dev):
```bash
pip install -r requirements-dev.txt
# or: make install
```

## How to Run

Clone:
```bash
git clone https://github.com/aarshdesai-ds/echonext-shd-detection.git
cd echonext-shd-detection
```

**Reproduce the model** — open the training notebook in `notebooks/` (Colab/Keras). It trains the ensemble and exports a versioned model bundle (`ens_seed*.keras`, `tabular_scaler.npz`, `metrics.json`).

**Serve locally** (after placing a model bundle in `./models/`):
```bash
make test        # ruff + pytest
make serve-all   # API + demo at http://localhost:8080  (/ demo, /docs API)
```

**Example request:**
```bash
curl -s localhost:8080/predict -H 'content-type: application/json' -d '{
  "waveform": [[...2500 rows × 12 leads...]],
  "tabular":  [1, 75, 70, 160, 100, 420, 65]
}'
# -> {"probability":0.78,"flag":true,"threshold":0.77,"risk_band":"high"}
```

## Results Summary

| Model | Test AUROC | Test AUPRC | Notes |
|---|---|---|---|
| Original 2D-CNN baseline | ~0.81* | — | validation only; ~48 ms receptive field |
| CNN (single, regularized) | 0.833 | 0.804 | corrected architecture |
| CNN + Transformer head | 0.830 | 0.802 | ablation — no significant gain |
| CNN + GRU head | 0.827 | 0.797 | ablation — no significant gain |
| **5-seed ensemble + TTA** | **0.842** | **0.812** | final model |
| EchoNext (paper)** | 0.852 | 0.785 | private 1.2M-ECG cohort, internal test |

\* original baseline reported validation AUROC only.  ** not head-to-head — see Key Finding 5.

## Ideas for Extension

1. **External validation.** Run the trained ensemble on a *second* public ECG dataset (PTB-XL, MIMIC-IV-ECG, Chapman-Shaoxing) mapped to the SHD label, and report — honestly — how much it degrades under distribution shift. The single highest-value addition.
2. **Self-supervised pretraining.** Pretrain the encoder with masked-ECG or contrastive objectives on unlabeled ECGs, then fine-tune. The most credible route to *beat* the benchmark, and where a Transformer could finally earn its keep.
3. **SHD subtype multi-label.** Predict the individual components (low EF, valvular disease, wall thickening, etc.) instead of a single composite label, and analyze per-subtype performance.
4. **Calibration & decision curves.** Add temperature scaling / isotonic calibration and decision-curve analysis to quantify net clinical benefit across thresholds.
5. **Interpretability.** Lead-level saliency / Grad-CAM over the ECG to show *where* the model looks, plus SHAP on the tabular branch.
6. **Fairness audit.** Break performance down by age and sex subgroups and test for disparities before any deployment claim.
7. **Model optimization for serving.** Export to ONNX / TFLite and distill the 5-model ensemble into a single network to cut latency and image size.
8. **Production hardening.** Add request logging, a `/metrics` endpoint, input-drift monitoring, and load testing to turn the demo into a monitored service.
9. **Threshold-free clinical framing.** Report sensitivity/specificity trade-offs at multiple operating points and the implied number-needed-to-screen at realistic prevalences.
10. **Architecture search.** Sweep depth/width, dilations, and structured state-space (S4/Mamba) backbones against the CNN baseline to test whether any long-range model meaningfully adds information.

---

*Data sourced from the EchoNext dataset on [PhysioNet](https://physionet.org/content/echonext/) (PhysioNet Restricted Health Data License). Method based on “Detecting structural heart disease from electrocardiograms using AI,” Nature (2025). Analysis covers the public EchoNext release; the public demo uses synthetic inputs only. Code: MIT.*
