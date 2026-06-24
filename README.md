# EchoNext-SHD 🫀 — structural heart disease detection from a 12-lead ECG

End-to-end ML system that detects **structural heart disease (SHD)** from a
standard 12-lead ECG. A 5-seed 1D-CNN ensemble (with clinical-feature fusion)
**matches** the published [EchoNext](https://www.nature.com/articles/s41586-025-09227-0)
baseline on AUROC and **beats** it on AUPRC — using ~16× less training data —
served on Google Cloud Run as a single container exposing both a JSON API and an
interactive Gradio demo.

> ⚠️ **Research/demo only. Not a medical device.** See [model_card.md](model_card.md).

| | This model (test, n=5442) | 95% CI | EchoNext paper |
|---|---|---|---|
| **AUROC** | 0.842 | [0.832, 0.853] | 0.852 |
| **AUPRC** | 0.812 | [0.797, 0.827] | 0.785 |
| **Brier** | 0.158 | — | — |
| Train size | ~72k ECGs | | 1.2M ECG–echo pairs |

🔗 **Live demo + API:** single Cloud Run service — demo at `/`, API at `/predict`, Swagger at `/docs` _(deploy in progress)_

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
  Colab training ──► deploy_bundle ──push──► gs://<bucket>/models/<version>/
                                                      │ pulled at build time
                                                      ▼
   GitHub Actions:  CI (ruff + pytest) ─► CD (docker build → Artifact Registry)
                                                      │
                                                      ▼
                       ┌──────────── GCP Cloud Run (scale-to-zero) ───────────┐
        ECG + 7        │  app.serve = FastAPI + Gradio (one container):        │
   clinical feats ──►  │    /  demo   ·   /predict API   ·   /healthz /info /docs│
                       │  shd.EnsemblePredictor → 5 models + TTA + tabular scaler│
                       └───────────────────────────────────────────────────────┘
```

## Repo layout
```
src/shd/        model.py (architecture + custom layers), infer.py (EnsemblePredictor),
                registry.py (versioned GCS model store)
app/api/        FastAPI service (pure API; importable/testable on its own)
app/demo/       Gradio app
app/serve.py    combined Cloud Run entrypoint: API + demo mounted at /
tests/          pytest (CI gate; model-dependent tests self-skip)
.github/        ci.yml (lint+test) · deploy.yml (GCS pull → build → Cloud Run)
infra/DEPLOY.md one-time GCP setup + cost guardrails
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

## Model registry
Trained ensembles are **versioned in Google Cloud Storage** (never committed) so
they can be reloaded without retraining — and it's the same store the Cloud Run
deploy pulls from at build time. [`src/shd/registry.py`](src/shd/registry.py)
manages versioned bundles (a durable local root, e.g. Google Drive, plus GCS).

```python
from shd.registry import ModelRegistry
BUCKET = "<your-gcs-bucket>"
reg = ModelRegistry("registry")
reg.pull_from_gcs("v1-cnn-ens5", BUCKET)   # fetch a version's bundle + index
predictor = reg.load_predictor("v1-cnn-ens5")   # no retraining
```
`reg.list()` shows every saved version with its metrics; `reg.best("AUPRC")`
selects the top one. (An optional Hugging Face Hub backend also exists.)

## Reproducibility
The runs are **seeded** (`tf.keras.utils.set_random_seed`), which controls weight
init, augmentation, and shuffling — giving statistical reproducibility (≈±0.003
AUROC across seeds) and deterministic ensemble diversity. It is **not** bit-exact
on GPU: mixed precision + non-deterministic cuDNN kernels add small jitter. The
real guarantee is the **versioned saved weights** — the exact artifact behind the
reported metrics is stored and reloadable. Each saved version records its seed,
library versions, and hyperparameters (`metrics.json → provenance`). For a
bit-exact reference run, enable `tf.config.experimental.enable_op_determinism()`
(slower, and disable mixed precision).

## Data
Trained on the public **[EchoNext dataset (PhysioNet)](https://physionet.org/content/echonext/)** —
100,000 de-identified 12-lead ECGs (250 Hz, `N×1×2500×12`) with
echocardiogram-confirmed structural-heart-disease labels and demographic/interval
metadata, from Columbia University Irving Medical Center. Access requires a
PhysioNet credentialed account and acceptance of the dataset's license/DUA.

The dataset is governed by the **PhysioNet Restricted Health Data License 1.5.0**
and is **not redistributed** in this repo. The public demo ships **no dataset
records** — it uses procedurally-generated **synthetic** ECGs ([`shd.synthetic`](src/shd/synthetic.py))
plus user uploads, so predictions there are illustrative only. To reproduce the
real metrics, obtain the data from PhysioNet (credentialed) and run the notebook.

**Citation:**
> EchoNext: A Dataset for Detecting Echocardiogram-Confirmed Structural Heart
> Disease from ECGs. PhysioNet. https://physionet.org/content/echonext/
>
> Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220, 2000.
>
> Underlying method: *Detecting structural heart disease from electrocardiograms
> using AI.* Nature (2025). https://www.nature.com/articles/s41586-025-09227-0

## License
Code: MIT (see [LICENSE](LICENSE)). Data: governed by the PhysioNet EchoNext
license/DUA — not covered by this repo's license.
