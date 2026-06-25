"""Hugging Face Spaces entry point — EchoNext-SHD demo (Gradio 5).

DUA-safe: inputs are synthetic or user-uploaded; no PhysioNet records ship.
Model weights are pulled at startup from a PRIVATE HF model repo using the
HF_TOKEN Space secret. The `shd` package is installed from GitHub (requirements.txt).
"""
import os

import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402
from shd.infer import N_LEADS, SEQ_LEN, TABULAR_ORDER, EnsemblePredictor  # noqa: E402
from shd.synthetic import synthetic_examples  # noqa: E402

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "aarshdesai04/echonext-shd-models")
VERSION = os.environ.get("MODEL_VERSION", "v1-cnn-ens5")
HF_TOKEN = os.environ.get("HF_TOKEN")

_root = snapshot_download(HF_MODEL_REPO, repo_type="model",
                          allow_patterns=[f"{VERSION}/*"], token=HF_TOKEN)
predictor = EnsemblePredictor(os.path.join(_root, VERSION))
EXAMPLES = synthetic_examples(3)
EX_NAMES = [e[0] for e in EXAMPLES]


def _plot(wave):
    fig, axes = plt.subplots(6, 2, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(axes.T.ravel()):
        ax.plot(wave[:, i], lw=0.6)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=14, fontsize=8)
        ax.set_yticks([]); ax.grid(alpha=0.2)
    fig.suptitle("12-lead ECG (synthetic / uploaded — 10 s)")
    fig.tight_layout()
    return fig


def _render(wave, tab):
    p = predictor.predict(wave, tab)
    return _plot(wave), {
        "SHD probability": round(p.probability, 3),
        "Decision (Se@90%Spec)": "FLAG for echo" if p.flag else "No flag",
        "Risk band": p.risk_band,
        "note": "illustrative — synthetic/user input, not validated",
    }


def run_example(name):
    _, w, t = EXAMPLES[EX_NAMES.index(name)]
    return _render(w, t)


def run_upload(file, sex, vrate, arate, pr, qrs, qtc, age):
    if file is None:
        raise gr.Error("Upload a .npy of shape (2500, 12) first.")
    path = file if isinstance(file, str) else file.name   # gradio 5 passes a path str
    try:
        return _render(np.load(path), [sex, vrate, arate, pr, qrs, qtc, age])
    except ValueError as e:
        raise gr.Error(str(e)) from e


with gr.Blocks(title="EchoNext-SHD") as demo:
    gr.Markdown(
        "# EchoNext-SHD — structural heart disease from a 12-lead ECG\n"
        "A 5-seed CNN ensemble. ⚠️ **Research/demo only — not a medical device.** "
        "Inputs are **synthetic** or user-uploaded; no patient data is shipped."
    )
    with gr.Tab("Synthetic examples"):
        sel = gr.Dropdown(EX_NAMES, value=EX_NAMES[0], label="Synthetic ECG")
        b1 = gr.Button("Predict", variant="primary")
        with gr.Row():
            p1 = gr.Plot(label="ECG"); o1 = gr.JSON(label="Prediction")
        b1.click(run_example, sel, [p1, o1])
        demo.load(run_example, sel, [p1, o1])
    with gr.Tab("Upload your own"):
        gr.Markdown(f"Upload a `.npy` ECG of shape **({SEQ_LEN}, {N_LEADS})**, z-scored.")
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
            p2 = gr.Plot(label="ECG"); o2 = gr.JSON(label="Prediction")
        b2.click(run_upload, [up, sex, vrate, arate, pr, qrs, qtc, age], [p2, o2])
    gr.Markdown(f"Feature order: `{TABULAR_ORDER}`")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
