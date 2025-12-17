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

# ---------------- AESTHETIC BACKGROUND ----------------
st.markdown("""
<style>

/* REMOVE TOP WHITE SPACE */
.block-container {
    padding-top: 0rem;
}

/* ---------- BODY & BACKGROUND ---------- */
.stApp {
    background: linear-gradient(rgba(245,255,250,0.85), rgba(245,255,250,0.85)),
                url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6')
                center/cover fixed;
}

/* ---------- HEADER ---------- */
.header {
    background: linear-gradient(135deg, #7ddc9c, #5bbf7a);
    padding: 25px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #0f3d2e;
    border-radius: 0 0 40px 40px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}

/* ---------- GLASS CARD STYLE ---------- */
.card, .result-box, .uses-box, .common {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(15px);
    border-radius: 30px;
    padding: 25px;
    box-shadow: 0 15px 30px rgba(0,0,0,0.15);
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background: linear-gradient(135deg,#81c784,#2e7d32);
    color: white;
    font-weight: bold;
    border-radius: 50px;
    padding: 10px 25px;
    transition: 0.3s;
}

/* ---------- RESULT IMAGE ---------- */
.result-box img {
    border-radius: 25px;
    border: 3px solid #c7f2dd;
}

/* ---------- COMMON PLANTS ---------- */
.common {
    text-align: center;
    color: #2e7d32;
    font-weight: bold;
    border: 2px solid #a5e6c6;
}

/* ---------- HEADINGS ---------- */
.section-title {
    color: #1e4f3b;
    font-size: 26px;
    font-weight: bold;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='header'>🌿 Herbal Plant Identification System</div>", unsafe_allow_html=True)

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

# ---------------- HERBAL USES ----------------
HERBAL_USES = {
    "Aloevera": ["Skin care", "Burn healing", "Digestive health"],
    "Neem": ["Blood purification", "Skin diseases"],
    "Tulasi": ["Cold & cough", "Immunity booster"],
    "Mint": ["Digestive aid", "Cold relief"],
}

CONFIDENCE_THRESHOLD = 0.75

# ---------------- UPLOAD SECTION ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload Plant Image", type=["jpg","jpeg","png"])
identify = st.button("✨ Identify Plant")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
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
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        if confidence < CONFIDENCE_THRESHOLD:
            st.error("❌ Not a Herbal / Medicinal Plant")
        else:
            st.success(f"🌱 {plant_name}")
            st.write(f"**Confidence:** {confidence:.2f}")

            st.markdown("<div class='uses-box'>", unsafe_allow_html=True)
            st.markdown("### 🌿 Medicinal Uses")
            for use in HERBAL_USES.get(plant_name, ["Data not available"]):
                st.markdown(f"- {use}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- COMMON PLANTS ----------------
st.markdown("<div class='section-title'>🌼 Common Medicinal Plants</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, name in zip([c1,c2,c3,c4], ["Tulasi","Neem","Mint","Aloevera"]):
    with col:
        st.markdown(f"<div class='common'><h4>{name}</h4></div>", unsafe_allow_html=True)
