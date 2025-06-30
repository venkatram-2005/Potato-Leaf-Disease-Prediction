import tensorflow as tf
import numpy as np
import streamlit as st
import os
import requests

# ✅ Class order matches training labels
class_names = ['Early_Blight', 'Healthy', 'Late_Blight']

# ✅ Hugging Face model URL
MODEL_URL = "https://huggingface.co/venkatram-2005/Potato-Leaf-Disease/resolve/main/potato_leaf_model.h5"

@st.cache_resource
def load_model():
    """
    Loads the model from local path or downloads it from Hugging Face if missing.
    Cached to avoid repeated reloads in Streamlit.
    """
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'potato_leaf_model.h5'))

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            response = requests.get(MODEL_URL)
            f.write(response.content)

    return tf.keras.models.load_model(model_path)

def load_model_and_predict(image_array):
    model = load_model()
    prediction = model.predict(image_array)
    return prediction, class_names

def build_functional_model_for_gradcam():
    """
    Builds a Functional model with weights copied from the original Sequential model.
    Works reliably with Grad-CAM even under Streamlit caching.
    """
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'potato_leaf_model.h5'))

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            response = requests.get(MODEL_URL)
            f.write(response.content)

    seq_model = tf.keras.models.load_model(model_path)

    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = inputs
    for layer in seq_model.layers:
        config = layer.get_config()
        new_layer = layer.__class__.from_config(config)
        x = new_layer(x)
        new_layer.set_weights(layer.get_weights())

    functional_model = tf.keras.Model(inputs=inputs, outputs=x)
    return functional_model
