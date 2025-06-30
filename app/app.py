import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
from PIL import Image

from model_loader import build_functional_model_for_gradcam, load_model_and_predict, load_model
from image_utils import preprocess_image
from grad_cam import make_gradcam_heatmap, overlay_gradcam

# ───────── Streamlit Page Setup ─────────
st.set_page_config(page_title="Potato Leaf Disease Detector", layout="centered")
st.title("🥔 Potato Leaf Disease Detector")

st.markdown("""
Potato plants are vulnerable to a variety of diseases that affect both yield and crop quality. 
Among the most damaging are **leaf diseases**, which manifest visually and can be identified through deep learning-based image classification.

This tool uses a trained CNN model to classify leaf images into three categories:

- 🌱 **Healthy** – Leaves showing no signs of disease.
- 🟠 **Early Blight** – Caused by *Alternaria solani*, presents as dark spots with concentric rings.
- 🔴 **Late Blight** – Caused by *Phytophthora infestans*, often appears as irregular water-soaked lesions that quickly spread.

Upload an image of a potato leaf, and the model will detect its condition and visualize how it made the decision using **Grad-CAM**.
""")

# ───────── Sidebar Info ─────────
with st.sidebar:
    st.title("🧠 Model Info")
    
    st.markdown("**Model Type:** Custom CNN (Sequential)\n\n"
                "**Input Shape:** 128 x 128 x 3\n\n"
                "**Trained On:** PLD 3-class dataset\n\n"
                "**Loss:** Categorical Crossentropy\n")

    st.markdown("---")
    st.subheader("📂 Classes")
    st.markdown("- Early Blight\n- Late Blight\n- Healthy")

    st.markdown("---")
    with st.expander("🕘 Most Recent Predictions"):
        if st.session_state.get("recent_preds"):
            for i, pred in enumerate(reversed(st.session_state.recent_preds[-5:]), 1):
                st.markdown(f"{i}. {pred}")
        else:
            st.markdown("No predictions yet.")

    st.markdown("---")
    st.caption("Built using TensorFlow, Streamlit, and Keras 🔥")

# ───────── File Upload ─────────
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    if st.button("Predict"):
        processed_img = preprocess_image(image, target_size=(128, 128))
        prediction, class_names = load_model_and_predict(processed_img)

        # Store in session state for persistence
        st.session_state.processed_img = processed_img
        st.session_state.prediction = prediction
        st.session_state.class_names = class_names

        st.session_state.recent_preds = st.session_state.get("recent_preds", [])
        st.session_state.recent_preds.append(
            f"{class_names[np.argmax(prediction)]} ({np.max(prediction)*100:.2f}%)"
        )

# ───────── Show Stored Prediction & Grad-CAM ─────────
if "prediction" in st.session_state:
    prediction = st.session_state.prediction
    class_names = st.session_state.class_names
    processed_img = st.session_state.processed_img

    st.markdown("---")
    st.success(f"🔍 **Prediction:** `{class_names[np.argmax(prediction)]}` "
               f"with `{np.max(prediction)*100:.2f}%` confidence")
    st.bar_chart(prediction)
    st.caption(f"Raw-Output: {prediction}")
