"""Gradio demo (public, DUA-safe).

Inputs are procedurally-generated SYNTHETIC ECGs (see `shd.synthetic`) or files
the user uploads — **no PhysioNet EchoNext data is shipped or redistributed**.
Predictions on synthetic/arbitrary inputs are illustrative only; the validated
AUROC/AUPRC live in the repo README (held-out dataset test set).
"""
from __future__ import annotations

import os
import sys

import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from shd.infer import N_LEADS, SEQ_LEN, TABULAR_ORDER, EnsemblePredictor  # noqa: E402
from shd.synthetic import synthetic_examples  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

predictor = EnsemblePredictor(MODEL_DIR)
EXAMPLES = synthetic_examples(3)
EX_NAMES = [e[0] for e in EXAMPLES]


def _plot_ecg(wave):
    fig, axes = plt.subplots(6, 2, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(axes.T.ravel()):
        ax.plot(wave[:, i], lw=0.6)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=14, fontsize=8)
        ax.set_yticks([])
        ax.grid(alpha=0.2)
    fig.suptitle("12-lead ECG (synthetic / uploaded — 10 s)", fontsize=12)
    fig.tight_layout()
    return fig


def _render(wave, tab):
    pred = predictor.predict(wave, tab)
    return _plot_ecg(wave), {
        "SHD probability": round(pred.probability, 3),
        "Decision (Se@90%Spec)": "FLAG for echo" if pred.flag else "No flag",
        "Risk band": pred.risk_band,
        "note": "illustrative — input is synthetic/user-provided, not validated",
    }


def run_example(name):
    _, wave, tab = EXAMPLES[EX_NAMES.index(name)]
    return _render(wave, tab)


def run_upload(file, sex, vrate, arate, pr, qrs, qtc, age):
    if file is None:
        raise gr.Error("Upload a .npy array of shape (2500, 12) first.")
    path = file if isinstance(file, str) else file.name   # gradio 5 passes a path str
    try:
        wave = np.load(path)
        return _render(wave, [sex, vrate, arate, pr, qrs, qtc, age])
    except ValueError as e:
        raise gr.Error(str(e)) from e


with gr.Blocks(title="EchoNext-SHD") as demo:
    gr.Markdown(
        "# EchoNext-SHD — structural heart disease from a 12-lead ECG\n"
        "A 5-seed CNN ensemble that **matches** the EchoNext AUROC (0.842 vs 0.852) "
        "and **beats** its AUPRC (0.812 vs 0.785) with ~16× less data.\n\n"
        "> ⚠️ Research/demo only — **not a medical device**. Demo inputs are "
        "**synthetic** (procedurally generated) or user-uploaded; no PhysioNet "
        "data is shipped. Predictions on these inputs are illustrative, not "
        "validated. Real metrics: see the repo README."
    )
    with gr.Tab("Synthetic examples"):
        sel = gr.Dropdown(EX_NAMES, value=EX_NAMES[0], label="Synthetic ECG")
        b1 = gr.Button("Predict", variant="primary")
        with gr.Row():
            p1 = gr.Plot(label="ECG")
            o1 = gr.JSON(label="Prediction")
        b1.click(run_example, sel, [p1, o1])
        demo.load(run_example, sel, [p1, o1])

    with gr.Tab("Upload your own"):
        gr.Markdown(f"Upload a `.npy` ECG of shape **({SEQ_LEN}, {N_LEADS})** "
                    f"(or {N_LEADS}×{SEQ_LEN}), z-scored. Set the clinical features:")
        up = gr.File(label=".npy ECG", file_types=[".npy"])
        with gr.Row():
            sex = gr.Number(label="sex (0/1)", value=1)
            vrate = gr.Number(label="ventricular_rate", value=72)
            arate = gr.Number(label="atrial_rate", value=72)
            pr = gr.Number(label="pr_interval", value=160)
        with gr.Row():
            qrs = gr.Number(label="qrs_duration", value=95)
            qtc = gr.Number(label="qt_corrected", value=420)
            age = gr.Number(label="age_at_ecg", value=60)
        b2 = gr.Button("Predict", variant="primary")
        with gr.Row():
            p2 = gr.Plot(label="ECG")
            o2 = gr.JSON(label="Prediction")
        b2.click(run_upload, [up, sex, vrate, arate, pr, qrs, qtc, age], [p2, o2])

    gr.Markdown(f"Feature order: `{TABULAR_ORDER}`")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
