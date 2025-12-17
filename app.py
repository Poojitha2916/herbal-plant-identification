import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Herbal Plant Identification",
    layout="wide"
)

# ---------------- CUSTOM AESTHETIC CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
}

.main-title {
    background: linear-gradient(135deg, #81c784, #66bb6a);
    padding: 25px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #1b5e20;
    border-radius: 0 0 40px 40px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}

.upload-box {
    background: #ffffff;
    padding: 40px;
    border-radius: 30px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.2);
    text-align: center;
}

.result-box {
    background: #f1fff8;
    padding: 35px;
    border-radius: 30px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
}

.card {
    background: #ffffff;
    padding: 22px;
    border-radius: 25px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌿 Herbal Plant Identification System</div>", unsafe_allow_html=True)
st.write("")

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

# ---------------- MEDICINAL USES (ALL CLASSES COVERED) ----------------
DEFAULT_USE = [
    "🌿 Traditional medicinal plant",
    "🧪 Used in herbal formulations",
    "🩺 Supports general wellness"
]

HERBAL_USES = {name: DEFAULT_USE for name in CLASS_NAMES}

# Specific enriched uses
HERBAL_USES.update({
    "Aloevera": ["Skin care", "Burn healing", "Digestive health"],
    "Neem": ["Blood purification", "Skin diseases", "Dental care"],
    "Tulasi": ["Cold & cough", "Boosts immunity", "Respiratory health"],
    "Mint": ["Improves digestion", "Cold relief", "Oral health"],
    "Amla": ["Vitamin C rich", "Hair health", "Immunity booster"],
    "Kunyit": ["Anti-inflammatory", "Wound healing"],
    "Jahe": ["Digestion", "Reduces nausea"],
    "Ashwagandha": ["Stress relief", "Improves strength"],
    "Brahmi": ["Improves memory", "Brain health"],
    "Lemon_grass": ["Stress reduction", "Digestive aid"],
    "Rose": ["Skin hydration", "Aromatherapy", "Stress relief"]
})

CONFIDENCE_THRESHOLD = 0.75

# ---------------- UI ----------------
st.write("")
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
st.markdown("### Upload Plant Image 🌸")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
identify = st.button("✨ Identify Plant")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RESULT ----------------
if identify and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    plant_name = CLASS_NAMES[class_id]

    st.write("")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        if confidence < CONFIDENCE_THRESHOLD:
            st.error("❌ Not a Herbal / Medicinal Plant")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.success(f"🌱 Identified Plant: {plant_name}")
            st.write(f"Confidence: {confidence:.2f}")
            st.markdown("### 🌿 Medicinal Uses")
            for u in HERBAL_USES[plant_name]:
                st.markdown(f"- {u}")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- COMMON PLANTS ----------------
st.write("")
st.markdown("## 🌼 Common Medicinal Plants")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("<div class='card'><b>Tulasi</b><br>Immunity booster</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><b>Neem</b><br>Skin & blood purifier</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><b>Aloevera</b><br>Skin & digestion</div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='card'><b>Mint</b><br>Digestive aid</div>", unsafe_allow_html=True)
