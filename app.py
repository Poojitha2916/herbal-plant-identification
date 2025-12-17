import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Herbal Plant Identification",
    layout="wide"
)

# ---------------- AESTHETIC BACKGROUND + RANDOM FLOATING LEAVES ----------------
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

/* ---------- RANDOM FLOATING LEAVES ---------- */
.leaf {
    position: fixed;
    font-size: 24px;
    animation-name: float;
    animation-iteration-count: infinite;
    animation-timing-function: linear;
    opacity: 0.3;
    z-index: 1;
}

@keyframes float {
    0% {top: -10%; transform: rotate(0deg);}
    100% {top: 110%; transform: rotate(360deg);}
}
""", unsafe_allow_html=True)

# Generate multiple leaves with random positions, sizes, and speeds
import random
leaves_html = ""
for i in range(20):
    left = random.randint(0, 95)
    size = random.randint(18, 32)
    duration = random.randint(15, 35)
    delay = random.randint(0, 15)
    leaves_html += f'<div class="leaf" style="left:{left}%; font-size:{size}px; animation-duration:{duration}s; animation-delay:{delay}s;">🍃</div>\n'

st.markdown(leaves_html, unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div style="
background: linear-gradient(135deg, #7ddc9c, #5bbf7a);
padding: 25px;
text-align: center;
font-size: 32px;
font-weight: bold;
color: #0f3d2e;
border-radius: 0 0 40px 40px;
box-shadow: 0 8px 20px rgba(0,0,0,0.25);
position: relative;
z-index: 2;">
🌿 Herbal Plant Identification System
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------- GLASS CARD STYLE ----------
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

# ---------- UPLOAD CARD ----------
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
    HERBAL_USES = { 
        "Aloevera": ["Skin care", "Burn healing", "Digestive health"], 
        "Neem": ["Blood purification", "Skin diseases"],
        "Tulasi": ["Cold & cough", "Immunity booster"],
        "Mint": ["Digestive aid", "Cold relief"],
    }
    for u in HERBAL_USES.get(plant_name, ["Data not available"]):
        st.markdown(f"- {u}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- COMMON PLANTS ----------
st.markdown("## 🌼 Common Medicinal Plants")
c1, c2, c3, c4 = st.columns(4)
for col, name in zip([c1,c2,c3,c4], ["Tulasi","Neem","Mint","Aloevera"]):
    with col:
        st.markdown(f"<div class='common'><h4>{name}</h4></div>", unsafe_allow_html=True)
