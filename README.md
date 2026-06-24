# EchoNext-SHD 🫀 — structural heart disease detection from a 12-lead ECG

End-to-end ML system that detects **structural heart disease (SHD)** from a
standard 12-lead ECG. A 5-seed 1D-CNN ensemble (with clinical-feature fusion)
**matches** the published [EchoNext](https://www.nature.com/articles/s41586-025-09227-0)
baseline on AUROC and **beats** it on AUPRC — using ~16× less training data —
served as a containerized API on Cloud Run with a live Gradio demo.

> ⚠️ **Research/demo only. Not a medical device.** See [model_card.md](model_card.md).

| | This model (test, n=5442) | 95% CI | EchoNext paper |
|---|---|---|---|
| **AUROC** | 0.842 | [0.832, 0.853] | 0.852 |
| **AUPRC** | 0.812 | [0.797, 0.827] | 0.785 |
| **Brier** | 0.158 | — | — |
| Train size | ~72k ECGs | | 1.2M ECG–echo pairs |

🔗 **Live demo:** _<HF Spaces link>_ · **API:** _<Cloud Run URL>_

## What's interesting here (engineering)
- **Diagnosed and fixed a broken baseline**: the original 2D-CNN used `(1,1)`
  kernels → a ~48 ms receptive field. Rebuilt as a deep `Conv1D` residual
  tokenizer (2500→79 tokens) + attention pooling + tabular fusion.
- **Honest ablation**: a Transformer/GRU temporal head was a *statistical wash*
  at 72k samples — reported, not hidden. The win came from receptive field,
  multimodal fusion, regularization, and a seed **ensemble + test-time augmentation**.
- **Rigorous eval**: bootstrap 95% CIs, calibration (Brier), and a
  sensitivity-at-fixed-specificity operating point chosen on validation.

## Architecture
```
                ┌─ Hugging Face Spaces ─┐      ┌──────── GCP Cloud Run ────────┐
 ECG + clinical │   Gradio demo (app/    │      │  FastAPI (app/api)             │
   features  ─► │   demo) → EnsemblePred │      │  /predict /healthz             │
                └────────────────────────┘      │  Docker, scale-to-zero         │
                          ▲                      └───────────────▲───────────────┘
                          │ in-process                           │ image + models
              ┌───────────┴───────────────────────────┐         │
              │ shd/ package: model.py, infer.py        │   GitHub Actions
              │ EnsemblePredictor (5 models + TTA + scaler)   CI (lint+test) → CD (build+deploy)
              └─────────────────────────────────────────┘   models pulled from HF Hub
```

## Repo layout
```
src/shd/        model.py (architecture + custom layers), infer.py (EnsemblePredictor)
app/api/        FastAPI service  (Cloud Run)
app/demo/       Gradio app       (HF Spaces)
tests/          pytest (CI gate; model-dependent tests self-skip)
.github/        ci.yml (lint+test) · deploy.yml (build+push+deploy)
infra/DEPLOY.md one-time GCP/HF setup + cost guardrails
Dockerfile · requirements*.txt · model_card.md · pyproject.toml · Makefile
```

## Quickstart
```bash
make install                 # dev deps
# drop the deploy bundle into ./models  (see infra/DEPLOY.md, step 0–1)
make test                    # lint + unit tests
make serve                   # API on http://localhost:8080  (/docs for Swagger)
make demo                    # Gradio UI on http://localhost:7860
make run                     # build + run the Docker image
```

### Example request
```bash
curl -s localhost:8080/predict -H 'content-type: application/json' -d '{
  "waveform": [[...2500 rows x 12 leads...]],
  "tabular":  [1, 75, 70, 160, 100, 420, 65]
}'
# -> {"probability":0.78,"flag":true,"threshold":0.77,"risk_band":"high"}
```

## Reproducing the model
Training lives in [`notebooks/`](notebooks/) (Colab/Keras). The export cell
writes the deploy bundle (models + scaler + sample ECGs + metrics) consumed by
this service.

## License & data
Code: MIT. The ECG dataset is **not** redistributed here; sample inputs are
de-identified arrays for demonstration only.
