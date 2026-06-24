# Deployment runbook (Google Cloud, free tier)

One target: a **single Cloud Run service** that serves both the prediction API
and the Gradio demo (`app.serve`), with model bundles stored in **Google Cloud
Storage**. No other platform required.

```
Colab training ──► deploy_bundle (models + scaler + samples + metrics)
                       │
                       ▼  push (versioned)
                 gs://<bucket>/models/<version>/
                       │
   GitHub Actions ─────┤ pull at build time → bake into image
        (CI → CD)      ▼
                 Cloud Run service  ┌ /          Gradio demo
                 (scale-to-zero)    ├ /predict   API
                                    ├ /healthz /info /docs
```

---

## 0. Produce the deploy bundle (once, from Colab)
Run the export cell (Cell 12). It writes `deploy_bundle/`:
```
ens_seed42.keras … ens_seed4.keras   # the 5 models
tabular_scaler.npz                    # mean/std of the 6 continuous features
sample_ecgs.npz                       # ~12 de-identified sample ECGs for the demo
metrics.json                          # AUROC/AUPRC/CIs + operating threshold
```

## 1. One-time GCP setup
```bash
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com artifactregistry.googleapis.com storage.googleapis.com

# Artifact Registry (Docker images)
gcloud artifacts repositories create echonext --repository-format=docker --location=<REGION>

# GCS bucket for the model registry
gcloud storage buckets create gs://<BUCKET> --location=<REGION>
```

## 2. Push the model bundle to GCS (versioned)
From Colab (Cell 13) or locally:
```bash
gcloud storage cp deploy_bundle/* gs://<BUCKET>/models/v1-cnn-ens5/
```
For local dev, also copy `deploy_bundle/*` into `echonext-shd/models/`.

## 3. Keyless GitHub → GCP auth (Workload Identity Federation)
```bash
gcloud iam workload-identity-pools create gh-pool --location=global
gcloud iam workload-identity-pools providers create-oidc gh-provider \
  --location=global --workload-identity-pool=gh-pool \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository"

gcloud iam service-accounts create gh-deployer
for ROLE in run.admin artifactregistry.writer storage.objectViewer iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member="serviceAccount:gh-deployer@<PROJECT_ID>.iam.gserviceaccount.com" \
    --role="roles/${ROLE}"
done
# bind the repo to impersonate the SA (see google-github-actions/auth docs)
```

## 4. GitHub repo configuration
**Variables** (Settings → Secrets and variables → Actions → Variables):
- `GCP_PROJECT_ID`, `GCP_REGION` (e.g. `us-central1`)
- `GCS_BUCKET` (your bucket name), `MODEL_VERSION` (e.g. `v1-cnn-ens5`)

**Secrets**:
- `GCP_WIF_PROVIDER` (full provider resource name)
- `GCP_DEPLOY_SA` (`gh-deployer@<PROJECT_ID>.iam.gserviceaccount.com`)

Push to `main` → CI runs → on success, deploy pulls the bundle from GCS, builds,
and ships. The Cloud Run URL is printed in the deploy step; open it for the demo,
`/predict` for the API, `/docs` for Swagger.

## 5. Cost guardrails (stay at $0)
- Cloud Run `--min-instances 0` → scales to zero; you pay only per request.
- `--max-instances 2`, `--concurrency 4`, `--memory 2Gi` cap worst-case spend.
- GCS Always Free: 5 GB-months (the bundle is ~170 MB); Artifact Registry storage
  is a few cents/month at most — set a budget alert if you want a hard guard.

## 6. Manual deploy (without GitHub Actions)
```bash
gcloud storage cp "gs://<BUCKET>/models/v1-cnn-ens5/*" models/
gcloud run deploy echonext-shd --source . --region <REGION> \
  --allow-unauthenticated --memory 2Gi --cpu 1 --min-instances 0 --max-instances 2
```
(`--source .` uses Cloud Build; the local `models/` is included in the build context.)
