"""Gradio demo for Hugging Face Spaces.

Loads the ensemble in-process and lets a visitor pick a de-identified sample
ECG, view the 12-lead strip, and see the model's SHD risk + decision.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow `from shd...` when running from repo root or Spaces
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from shd.infer import EnsemblePredictor, TABULAR_ORDER  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

predictor = EnsemblePredictor(MODEL_DIR)
samples = np.load(os.path.join(MODEL_DIR, "sample_ecgs.npz"), allow_pickle=True)
SAMPLE_WAVES = samples["waveforms"]          # (n, 2500, 12)
SAMPLE_TAB = samples["tabular_raw"]          # (n, 7)
SAMPLE_Y = samples["labels"]                 # (n,)
CHOICES = [f"Patient {i+1} (true label: {'SHD' if y else 'normal'})"
           for i, y in enumerate(SAMPLE_Y)]


def _plot_ecg(wave):
    fig, axes = plt.subplots(6, 2, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(axes.T.ravel()):
        ax.plot(wave[:, i], lw=0.6)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=14, fontsize=8)
        ax.set_yticks([]); ax.grid(alpha=0.2)
    fig.suptitle("12-lead ECG (10 s)", fontsize=12)
    fig.tight_layout()
    return fig


def run(choice):
    idx = CHOICES.index(choice)
    wave, tab, y = SAMPLE_WAVES[idx], SAMPLE_TAB[idx], int(SAMPLE_Y[idx])
    pred = predictor.predict(wave, tab)
    label = {
        "SHD probability": round(pred.probability, 3),
        "Decision (Se@90%Spec)": "FLAG for echo" if pred.flag else "No flag",
        "Risk band": pred.risk_band,
        "Ground truth": "SHD" if y else "normal",
    }
    tab_str = "\n".join(f"- {k}: {v:g}" for k, v in zip(TABULAR_ORDER, tab))
    return _plot_ecg(wave), label, tab_str


with gr.Blocks(title="EchoNext-SHD") as demo:
    gr.Markdown(
        "# EchoNext-SHD — structural heart disease from a 12-lead ECG\n"
        "A 5-seed CNN ensemble that **matches** the published EchoNext AUROC "
        "(0.842 vs 0.852) and **beats** its AUPRC (0.812 vs 0.785) with ~16x "
        "less data. Pick a sample patient to see a prediction.\n\n"
        "> ⚠️ Research/demo only. Not a medical device. Inputs are de-identified samples."
    )
    with gr.Row():
        sel = gr.Dropdown(CHOICES, value=CHOICES[0], label="Sample patient")
        btn = gr.Button("Predict", variant="primary")
    with gr.Row():
        plot = gr.Plot(label="ECG")
        with gr.Column():
            out = gr.Label(label="Prediction")
            feats = gr.Textbox(label="Clinical features", lines=8)
    btn.click(run, inputs=sel, outputs=[plot, out, feats])
    demo.load(run, inputs=sel, outputs=[plot, out, feats])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
