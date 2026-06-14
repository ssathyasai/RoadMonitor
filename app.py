import streamlit as st
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

from PIL import Image
import tensorflow as tf


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RoadMonitor - AI Road Damage Detection",
    layout="wide"
)


# =========================================================
# LOAD CSS  (robust path: works regardless of working dir)
# =========================================================

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<h1 class='main-title'>RoadMonitor - AI Road Damage Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 class='subtitle'>Smart City Infrastructure Monitoring using CNN</h3>",
    unsafe_allow_html=True
)
st.markdown("---")


# =========================================================
# LOAD LABEL MAPPING
# =========================================================

def load_label_mapping():
    mapping_path = os.path.join(os.path.dirname(__file__), "label_mapping.json")
    with open(mapping_path, "r") as f:
        return json.load(f)

label_mapping = load_label_mapping()
# Sort labels by their integer index so list order matches model output
labels = sorted(label_mapping.keys(), key=lambda k: label_mapping[k])


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "road_damage_cnn_model.keras")
    if not os.path.exists(model_path):
        st.error("Model file not found. Please train the model first:")
        st.code("python train_model.py")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()


# =========================================================
# IMAGE SIZE  (must match training configuration)
# =========================================================

IMG_SIZE = 224   # MobileNetV2


# =========================================================
# SEVERITY: based on damage TYPE, not just confidence
# =========================================================

SEVERITY_BY_TYPE = {
    "potholes": "HIGH",
    "cracks":   "MEDIUM",
    "manholes": "LOW",
}


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.header("📤 Upload Road Image")
st.write("""
Upload a road surface image to detect:
- **Potholes** — high risk, immediate repair needed
- **Cracks** — medium risk, preventive maintenance
- **Manholes** — low risk, routine inspection
""")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Drag and drop supported"
)


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    st.success(f"Image uploaded: **{uploaded_file.name}**")

    # ----- Preview -----
    st.header("Uploaded Image Preview")
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Road Image", width=500)

    # ----- Preprocessing (MobileNetV2: scale to [-1, 1]) -----
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array  = np.array(img_resized, dtype=np.float32)
    img_array  = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array  = np.expand_dims(img_array, axis=0)   # shape: (1, 224, 224, 3)

    # ----- Prediction -----
    if model is not None:
        try:
            predictions  = model.predict(img_array)
            probabilities = predictions[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()
    else:
        st.warning("Model not loaded — cannot make predictions.")
        st.stop()

    predicted_index      = int(np.argmax(probabilities))
    confidence           = float(np.max(probabilities))
    predicted_label      = labels[predicted_index]
    confidence_pct       = confidence * 100

    # Severity is determined by damage type (not confidence level)
    severity = SEVERITY_BY_TYPE.get(predicted_label, "UNKNOWN")

    # ----- Results -----
    st.header("Prediction Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"**Detected:** {predicted_label.upper()}")
    with col2:
        st.info(f"**Confidence:** {confidence_pct:.2f}%")
    with col3:
        if severity == "HIGH":
            st.error(f"**Severity:** {severity}")
        elif severity == "MEDIUM":
            st.warning(f"**Severity:** {severity}")
        else:
            st.info(f"**Severity:** {severity}")

    # ----- Confidence bar -----
    st.subheader("Prediction Confidence")
    st.progress(confidence)

    # ----- Bar chart -----
    st.header("Prediction Visualization")
    df = pd.DataFrame({"Class": labels, "Probability": probabilities})

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e53935" if lbl == predicted_label else "#90caf9" for lbl in labels]
    ax.bar(df["Class"], df["Probability"], color=colors)
    ax.set_title("Class Confidence")
    ax.set_xlabel("Damage Type")
    ax.set_ylabel("Confidence Score")
    ax.set_ylim(0, 1)
    for i, prob in enumerate(probabilities):
        ax.text(i, prob + 0.01, f"{prob:.2%}", ha='center', fontsize=10)
    st.pyplot(fig)

    # ----- Probability table -----
    st.subheader("Class Probabilities")
    for lbl, prob in zip(labels, probabilities):
        st.write(f"**{lbl}** : {prob:.4f}")

    # ----- Maintenance Recommendations -----
    st.header("Maintenance Recommendations")

    if predicted_label == "potholes":
        st.error("""
**🔴 Immediate maintenance required.**

Potholes present a high-risk condition:
- Vehicle tyre damage / punctures
- Loss of vehicle control
- Risk of traffic accidents

**Priority Level: HIGH — Schedule urgent repair.**
""")

    elif predicted_label == "cracks":
        st.warning("""
**🟡 Preventive maintenance recommended.**

Surface cracks can worsen over time due to weather and traffic load.
Early intervention prevents more costly repairs later.

**Priority Level: MEDIUM — Schedule within 30 days.**
""")

    elif predicted_label == "manholes":
        st.info("""
**🔵 Routine inspection recommended.**

Manhole cover detected. Verify:
- Cover is flush with road surface
- No settlement or cracking around frame

**Priority Level: LOW — Include in next scheduled inspection.**
""")

    # ----- Summary -----
    st.markdown("---")
    st.subheader("CNN Analysis Summary")
    st.write(f"""
The CNN model analysed the uploaded image through 4 convolutional blocks,
extracting features at increasing levels of abstraction (edges → textures → patterns → shapes).

- **Predicted class:** {predicted_label.upper()}
- **Confidence:** {confidence_pct:.2f}%
- **Severity:** {severity}

A higher confidence score indicates that the detected visual features strongly
match the training examples for that damage category.
""")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    "<center><h4>RoadMonitor — AI-Powered Smart City Infrastructure Monitoring</h4></center>",
    unsafe_allow_html=True
)
