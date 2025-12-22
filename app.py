import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Herbal Plant Identification", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("herbal_model.h5")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

st.title("🌿 Herbal Plant Identification")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    confidence = np.max(preds)
    plant_name = class_names[np.argmax(preds)]

    if confidence < 0.70:
        st.error("❌ Given plant is NOT a Herbal Plant")
    else:
        st.success(f"🌱 Plant Name: {plant_name}")
        st.write(f"Confidence: {confidence:.2f}")
