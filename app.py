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

# ---------------- AESTHETIC BACKGROUND + FLOATING LEAVES ----------------
st.markdown("""
<style>
/* ---------- BODY & BACKGROUND ---------- */
.stApp {
    background: linear-gradient(rgba(245,255,250,0.85), rgba(245,255,250,0.85)),
                url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6') center/cover fixed;
    background-size: cover;
    position: relative;
    overflow-x: hidden;
}

/* ---------- FLOATING LEAVES ---------- */
.leaf {
    position: fixed;
    font-size: 28px;
    animation: float 20s linear infinite;
    opacity: 0.3;
    z-index: 1;
}
.l1 { left: 10%; animation-delay: 0s; }
.l2 { left: 40%; animation-delay: 7s; }
.l3 { left: 70%; animation-delay: 12s; }
.l4 { left: 85%; animation-delay: 5s; }

@keyframes float {
    0% {top: -10%; transform: rotate(0deg);}
    100% {top: 110%; transform: rotate(360deg);}
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
    position: relative;
    z-index: 2;
}

/* ---------- GLASS CARD STYLE ---------- */
.card, .result-box, .common {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(15px);
    border-radius: 30px;
    padding: 25px;
    box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    transition: 0.3s ease;
    position: relative;
    z-index: 2;
}

.card:hover, .result-box:hover, .common:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.25);
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background: linear-gradient(135deg,#81c784,#2e7d32);
    color: white;
    font-weight: bold;
    border-radius: 50px;
    padding: 10px 25px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
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
    padding: 20px;
    border: 2px solid #a5e6c6;
    border-radius: 25px;
    background: rgba(255,255,255,0.6);
}

/* ---------- HEADINGS ---------- */
h2, h4 {
    color: #1e4f3b;
}
</style>

<!-- FLOATING LEAVES ELEMENTS -->
<div class="leaf l1">🍃</div>
<div class="leaf l2">🍃</div>
<div class="leaf l3">🍃</div>
<div class="leaf l4">🍃</div>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='header'>🌿 Herbal Plant Identification System</div>", unsafe_allow_html=True)
st.write("")

# ---------------- MODEL & DATA ----------------
MODEL_PATH = "herbal_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        # Replace with actual gdown download if needed
        pass
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

# model = load_model()  # Uncomment when model available

CLASS_NAMES = ["Aloevera","Neem","Tulasi","Mint"]
HERBAL_USES = { 
    "Aloevera": ["Skin care", "Burn healing", "Digestive health"], 
    "Neem": ["Blood purification", "Skin diseases"],
    "Tulasi": ["Cold & cough", "Immunity booster"],
    "Mint": ["Digestive aid", "Cold relief"],
}

# ---------------- UI FLOW ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Plant Image 🌸", type=["jpg","jpeg","png"])
identify = st.button("✨ Identify Plant")
st.markdown("</div>", unsafe_allow_html=True)

if identify and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)
    plant_name = "Neem"  # Example prediction
    confidence = 0.92

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.success(f"🌱 {plant_name}")
    st.write(f"Confidence: {confidence:.2f}")
    st.markdown("### 🌿 Medicinal Uses")
    for u in HERBAL_USES.get(plant_name, ["Data not available"]):
        st.markdown(f"- {u}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- COMMON PLANTS ----------------
st.markdown("## 🌼 Common Medicinal Plants")
c1, c2, c3, c4 = st.columns(4)
for col, name in zip([c1,c2,c3,c4], ["Tulasi","Neem","Mint","Aloevera"]):
    with col:
        st.markdown(f"<div class='common'><h4>{name}</h4></div>", unsafe_allow_html=True)
