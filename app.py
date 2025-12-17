import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image

MODEL_URL = "https://drive.google.com/uc?id=1a-_536wX34s8nakc84eI6TPatxmidKva"
MODEL_PATH = "herbal_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first time only)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

st.title("🌿 Herbal Plant Identification")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    confidence = float(np.max(predictions))

    st.success(f"Prediction class ID: {class_id}")
    st.info(f"Confidence: {confidence:.2f}")
