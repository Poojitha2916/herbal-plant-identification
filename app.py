import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import gdown
import os

st.set_page_config(page_title="Herbal Plant Identification", layout="wide")

# ---------------- MODEL ----------------
MODEL_URL = "https://drive.google.com/file/d/1a-_536wX34s8nakc84eI6TPatxmidKva/view"
MODEL_PATH = "herbal_model.h5"

CLASS_URL = "https://drive.google.com/uc?id=YOUR_CLASS_JSON_ID"
CLASS_PATH = "class_names.json"

@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True)

    if not os.path.exists(CLASS_PATH):
        gdown.download(CLASS_URL, CLASS_PATH)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    return model, class_names

model, class_names = load_model_and_classes()

# ---------------- HERBAL DATABASE ----------------
HERBAL_USES = {
    "mint": ["Improves digestion", "Relieves cold", "Oral health"],
    "neem": ["Skin diseases", "Blood purification", "Dental care"],
    "tulsi": ["Cold & cough", "Immunity booster"],
    "aloe": ["Skin care", "Digestion"],
}

# ---------------- UI ----------------
st.title("🌿 Herbal Plant Identification")

uploaded_file = st.file_uploader("Upload plant image", type=["jpg","jpeg","png"])
identify = st.button("✨ Identify Plant")

if identify and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = str(np.argmax(preds))
    confidence = float(np.max(preds))

    plant_name = class_names[class_id].lower()

    if plant_name in HERBAL_USES:
        st.success(f"🌱 Plant Identified: {plant_name.title()}")
        st.write(f"Confidence: {confidence:.2f}")
        st.markdown("### 🌿 Medicinal Uses")
        for u in HERBAL_USES[plant_name]:
            st.markdown(f"- {u}")
    else:
        st.error("❌ This is NOT a Herbal / Medicinal Plant")

# ---------------- COMMON PLANTS ----------------
st.markdown("## 🌼 Common Medicinal Plants")
st.write("Tulsi • Neem • Mint • Aloe Vera • Turmeric")
