---
title: EchoNext SHD Demo
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# EchoNext-SHD — demo

Structural heart disease detection from a 12-lead ECG (5-seed CNN ensemble).

⚠️ **Research/demo only — not a medical device.** Inputs are synthetic or
user-uploaded; no patient data is shipped. Code & validated metrics:
https://github.com/aarshdesai-ds/echonext-shd-detection

Model weights are loaded at runtime from a private HF model repo via the
`HF_TOKEN` Space secret.
