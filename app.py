import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Road Damage Detection",
    layout="wide"
)

# =====================================
# LOAD CSS
# =====================================

with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================

model = load_model(os.path.join(os.path.dirname(__file__), "road_damage_cnn_model.keras"))

# =====================================
# CLASS LABELS
# =====================================

class_names = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

# =====================================
# HEADER
# =====================================

st.markdown("""
<div class="hero">

<h1>🚧 AI-Based Road Damage Detection System</h1>

<p>
Smart City Infrastructure Monitoring using CNN
</p>

</div>
""", unsafe_allow_html=True)

# =====================================
# ABOUT SECTION
# =====================================

st.markdown("""
<div class="glass">

<h2>📌 About the Project</h2>

<ul>
<li>Road monitoring helps prevent accidents and improve public safety.</li>

<li>Potholes and cracks can damage vehicles and increase traffic risks.</li>

<li>CNN models automatically analyze road surface images using deep learning.</li>

<li>AI-based road inspection is used in smart cities and infrastructure monitoring.</li>
</ul>

</div>
""", unsafe_allow_html=True)

# =====================================
# UPLOAD SECTION
# =====================================

st.markdown("""
<div class="glass">

<h2>📤 Upload Road Image</h2>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1.3, 1])

    # =================================
    # LEFT COLUMN
    # =================================

    with col1:

        st.markdown("""
        <div class="glass">
        <h2>🖼 Uploaded Image</h2>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

    # =================================
    # PREPROCESSING
    # =================================

    img = np.array(image)

    if len(img.shape) == 3 and img.shape[-1] == 4:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2RGB
        )

    img = cv2.resize(img, (224,224))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # =================================
    # PREDICTION
    # =================================

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    predicted_label = class_names[predicted_class]

    # =================================
    # SEVERITY
    # =================================

    if confidence > 85:

        severity = "HIGH"

    elif confidence > 60:

        severity = "MEDIUM"

    else:

        severity = "LOW"

    # =================================
    # RIGHT COLUMN
    # =================================

    with col2:

        st.markdown("""
        <div class="glass">
        <h2>📊 Prediction Result</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <h3>Detected Damage</h3>
            <h1>{predicted_label}</h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <h3>Confidence</h3>
            <h1>{confidence:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <h3>Severity</h3>
            <h1>{severity}</h1>
        </div>
        """, unsafe_allow_html=True)

    # =================================
    # CHART
    # =================================

    st.markdown("""
    <div class="glass">
    <h2>📈 Confidence Visualization</h2>
    </div>
    """, unsafe_allow_html=True)

    probs = prediction[0] * 100

    fig, ax = plt.subplots(figsize=(7,3))

    bars = ax.bar(
        list(class_names.values()),
        probs
    )

    fig.patch.set_facecolor("#0f172a")

    ax.set_facecolor("#0f172a")

    ax.tick_params(colors="white")

    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    ax.set_ylabel(
        "Confidence %",
        color="white"
    )

    ax.set_title(
        "Prediction Probability",
        color="white"
    )

    st.pyplot(fig)

    # =================================
    # RECOMMENDATION
    # =================================

    st.markdown("""
    <div class="glass">
    <h2>⚠ Recommendation</h2>
    </div>
    """, unsafe_allow_html=True)

    if predicted_label == "Pothole":

        st.error(
            "Immediate maintenance recommended. High-risk road condition detected."
        )

    elif predicted_label == "Crack":

        st.warning(
            "Preventive repair is advised to avoid future road deterioration."
        )

    else:

        st.info(
            "Ensure proper alignment and secure manhole covering."
        )