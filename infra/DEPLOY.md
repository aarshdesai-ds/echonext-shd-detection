# Deployment runbook

Two targets, both free-tier:
1. **Hugging Face Spaces** — the clickable Gradio demo (put this link on your résumé).
2. **Google Cloud Run** — the production FastAPI, deployed by GitHub Actions.

The trained ensemble lives on the **Hugging Face Hub** (free model registry) and
is pulled at build/deploy time — it is never committed to git.

---

## 0. Produce the deploy bundle (once, from Colab)
Run the export cell (Cell 12 in the training notebook). It writes `deploy_bundle/`:
```
ens_seed42.keras ... ens_seed4.keras   # the 5 models
tabular_scaler.npz                     # mean/std of the 6 continuous features
sample_ecgs.npz                        # ~12 de-identified sample ECGs for the demo
metrics.json                           # AUROC/AUPRC/CIs + operating threshold
```

## 1. Push the bundle to the Hugging Face Hub
```bash
pip install "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli repo create echonext-shd-models --type model
huggingface-cli upload echonext-shd-models ./deploy_bundle . --repo-type model
```
For local dev, also copy `deploy_bundle/*` into `echonext-shd/models/`.

## 2. Hugging Face Spaces (demo)
- Create a **Gradio** Space.
- Set the Space to pull the model repo, or add it as a submodule; simplest is to
  set `MODEL_DIR` and `huggingface-cli download` in the Space, or upload the
  bundle into the Space's `models/`.
- App entrypoint: `app/demo/app.py`; requirements: `requirements-demo.txt`.

## 3. Google Cloud Run (API) — one-time GCP setup
```bash
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create echonext --repository-format=docker --location=<REGION>
```
### Workload Identity Federation (keyless GitHub auth — recommended)
```bash
# pool + provider
gcloud iam workload-identity-pools create gh-pool --location=global
gcloud iam workload-identity-pools providers create-oidc gh-provider \
  --location=global --workload-identity-pool=gh-pool \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository"
# deploy service account + roles
gcloud iam service-accounts create gh-deployer
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:gh-deployer@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/run.admin"
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:gh-deployer@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
# allow the repo to impersonate the SA (see GitHub docs for the full binding)
```

### GitHub repo configuration
**Variables** (Settings → Secrets and variables → Actions → Variables):
- `GCP_PROJECT_ID`, `GCP_REGION` (e.g. `us-central1`), `HF_MODEL_REPO` (e.g. `yourname/echonext-shd-models`)

**Secrets**:
- `GCP_WIF_PROVIDER` (full provider resource name), `GCP_DEPLOY_SA` (SA email)
- `HF_TOKEN` (only if the model repo is private)

Push to `main` → CI runs → on success, deploy runs → Cloud Run URL is printed.

## 4. Cost guardrails (stay at $0)
- Cloud Run `--min-instances 0` → scales to zero, no idle cost.
- `--max-instances 2`, `--concurrency 4`, `--memory 2Gi` cap worst-case spend.
- HF Spaces (CPU basic) and GitHub Actions (public repo) are free.
