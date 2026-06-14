# RoadMonitor

AI-powered road damage detection using MobileNetV2 and Streamlit.

🚀 **Live Demo:** [roadmonitor-143.streamlit.app](https://roadmonitor-143.streamlit.app/)

---

## What it does

Upload a road image and get:
- Damage type: **Pothole**, **Crack**, or **Manhole**
- Confidence score
- Severity level and maintenance recommendation

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

MobileNetV2 transfer learning — input 224×224, 3 output classes, ~64% validation accuracy.
