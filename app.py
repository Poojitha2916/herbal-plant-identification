import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Herbal Plant Identification", layout="wide")

# ---------------- MODEL ----------------
MODEL_URL = "https://drive.google.com/file/d/1a-_536wX34s8nakc84eI6TPatxmidKva/view"
MODEL_PATH = "herbal_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- CLASS NAMES ----------------
CLASS_NAMES = [
    "Adas","Aloevera","Amla","Amruta_Balli","Andong Merah","Arali",
    "Ashoka","Ashwagandha","Avacado","Bamboo","Basale","Belimbing Wulu",
    "Beluntas","Betadin","Betel","Betel_Nut","Brahmi","Castor",
    "Cincau Perdu","Curry_Leaf","Daun Afrika","Daun Cabe Jawa",
    "Daun Cocor Bebek","Daun Kumis Kucing","Daun Mangkokan","Daun Suji",
    "Daun Ungu","Dewa Ndaru","Doddapatre","Ekka","Gandarusa","Ganike",
    "Garut","Gauva","Geranium","Henna","Hibiscus","Honge","Honje","Iler",
    "Insulin","Jahe","Jasmine","Jeruk Nipis","Kapulaga","Kayu Putih",
    "Kecibling","Kemangi","Kembang Sepatu","Kenanga","Kunyit","Lampes",
    "Legundi","Lemon","Lemon_grass","Lidah Buaya","Mahkota Dewa","Mango",
    "Melati","Meniran","Mint","Murbey","Nagadali","Neem","Nilam",
    "Nithyapushpa","Nooni","Pacing Petul","Pandan","Pappaya",
    "Patah Tulang","Pecut Kuda","Pepper","Pomegranate","Raktachandini",
    "Rose","Saga Manis","Sapota","Secang","Sereh","Sirih","Srikaya",
    "Tin","Tulasi","Wood_sorel","Zigzag"
]

HERBAL_USES = {
    "Aloevera": ["Skin care", "Burn healing", "Digestive health"],
    "Neem": ["Blood purification", "Skin diseases", "Dental care"],
    "Tulasi": ["Cold & cough", "Immunity booster", "Respiratory health"],
    "Mint": ["Digestion", "Cold relief", "Oral health"],
    "Amla": ["Vitamin C rich", "Boosts immunity", "Hair health"],
    "Kunyit": ["Anti-inflammatory", "Wound healing"],
    "Jahe": ["Digestion", "Reduces nausea"]
}

# ---------------- UI ----------------
st.markdown("## 🌿 Herbal Plant Identification System")

uploaded_file = st.file_uploader("Upload plant image", type=["jpg","jpeg","png"])
identify = st.button("✨ Identify Plant")

if identify and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    plant_name = CLASS_NAMES[class_id]

    col1, col2 = st.columns([1,2])
    with col1:
        st.image(image, use_column_width=True)

    with col2:
        if plant_name in HERBAL_USES:
            st.success(f"🌱 Identified Plant: {plant_name}")
            st.write(f"Confidence: {confidence:.2f}")
            st.markdown("### 🌿 Medicinal Uses")
            for u in HERBAL_USES[plant_name]:
                st.markdown(f"- {u}")
        else:
            st.error("❌ This is NOT a Herbal / Medicinal Plant")

# ---------------- COMMON PLANTS ----------------
st.markdown("## 🌼 Common Medicinal Plants")
st.write("Tulasi • Neem • Mint • Aloevera • Amla • Kunyit")
