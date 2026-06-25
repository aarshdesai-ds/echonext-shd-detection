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
