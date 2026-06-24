"""Versioned model registry — persist trained ensembles and reload them without
retraining.

Two backends, same layout:
* a local root (e.g. a Google Drive folder) — durable across Colab restarts;
* the Hugging Face Hub (optional) — versioned, machine-independent reuse, and
  the same place the Cloud Run deploy pulls from.

Layout::

    <root>/
        registry.json                 # index: version -> {metrics, created, files}
        <version>/                     # one self-contained bundle per version
            ens_seed*.keras
            tabular_scaler.npz
            sample_ecgs.npz
            metrics.json

A "version" is any label you choose, e.g. ``v1-cnn-ensemble`` or a date.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Optional

INDEX_NAME = "registry.json"
BUNDLE_GLOBS = ("ens_seed*.keras", "tabular_scaler.npz", "sample_ecgs.npz", "metrics.json")


def _flatten_metric(m):
    """Accept either a float or the {'value': ...} shape used in metrics.json."""
    return m["value"] if isinstance(m, dict) and "value" in m else m


class ModelRegistry:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, INDEX_NAME)

    # ── index ────────────────────────────────────────────────────────────────
    def _load_index(self) -> dict:
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self, idx: dict) -> None:
        with open(self.index_path, "w") as f:
            json.dump(idx, f, indent=2)

    def list(self) -> dict:
        return self._load_index()

    def path(self, version: str) -> str:
        return os.path.join(self.root, version)

    # ── save ───────────────────────────────────────────────────────────────────
    def save(self, version: str, src_dir: str, metrics: Optional[dict] = None,
             overwrite: bool = False) -> str:
        """Copy a bundle (the files matching BUNDLE_GLOBS) from ``src_dir`` into
        ``<root>/<version>`` and record it in the index."""
        import glob

        dst = self.path(version)
        if os.path.exists(dst) and not overwrite:
            raise FileExistsError(f"version {version!r} exists; pass overwrite=True")
        os.makedirs(dst, exist_ok=True)

        files = []
        for pat in BUNDLE_GLOBS:
            for fp in glob.glob(os.path.join(src_dir, pat)):
                shutil.copy(fp, os.path.join(dst, os.path.basename(fp)))
                files.append(os.path.basename(fp))
        if not files:
            raise FileNotFoundError(f"no bundle files matched in {src_dir!r}")

        if metrics is None and "metrics.json" in files:
            with open(os.path.join(dst, "metrics.json")) as f:
                metrics = json.load(f)

        idx = self._load_index()
        idx[version] = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "files": sorted(files),
            "metrics": metrics or {},
        }
        self._save_index(idx)
        return dst

    # ── select ─────────────────────────────────────────────────────────────────
    def best(self, metric: str = "AUPRC", higher_is_better: bool = True) -> str:
        idx = self._load_index()
        if not idx:
            raise ValueError("registry is empty")
        scored = [(v, _flatten_metric(meta.get("metrics", {}).get(metric)))
                  for v, meta in idx.items()]
        scored = [(v, s) for v, s in scored if isinstance(s, (int, float))]
        if not scored:
            raise ValueError(f"no versions have numeric metric {metric!r}")
        return (max if higher_is_better else min)(scored, key=lambda x: x[1])[0]

    def load_predictor(self, version: Optional[str] = None, **kw):
        """Reload an ensemble for inference — no training. Defaults to best AUPRC."""
        from .infer import EnsemblePredictor
        version = version or self.best()
        return EnsemblePredictor(self.path(version), **kw)

    # ── Hugging Face Hub (optional) ─────────────────────────────────────────────
    def push_to_hf(self, version: str, repo_id: str, private: bool = True,
                   token: Optional[str] = None) -> str:
        from huggingface_hub import HfApi, create_repo
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)
        api = HfApi()
        api.upload_folder(folder_path=self.path(version), path_in_repo=version,
                          repo_id=repo_id, repo_type="model", token=token)
        # keep a shared index at the repo root
        api.upload_file(path_or_fileobj=self.index_path, path_in_repo=INDEX_NAME,
                        repo_id=repo_id, repo_type="model", token=token)
        return f"https://huggingface.co/{repo_id}/tree/main/{version}"

    def pull_from_hf(self, version: str, repo_id: str, token: Optional[str] = None) -> str:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, repo_type="model", local_dir=self.root,
                          allow_patterns=[f"{version}/*", INDEX_NAME], token=token)
        return self.path(version)
