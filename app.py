import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Herbal Plant Identification", layout="centered")

# Load trained model
model = tf.keras.models.load_model("herbal_model.h5")

# Class labels (CHANGE according to your dataset)
CLASS_NAMES = ['AloeVera', 'Neem', 'Tulsi']

st.title("🌿 Herbal Plant Identification System")
st.write("Upload a leaf image to identify the herbal plant")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    st.success(f"🌱 Predicted Herbal Plant: **{predicted_class}**")

