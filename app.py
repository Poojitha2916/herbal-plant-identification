import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Herbal Plant Identification", layout="wide")

# ---------------- MODEL ----------------
MODEL_URL = "https://drive.google.com/file/d/1a-_536wX34s8nakc84eI6TPatxmidKva/view"
MODEL_PATH = "herbal_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- PLANT DATABASE ----------------
PLANT_INFO = {
    60: {
        "name": "Mint (Pudina)",
        "uses": [
            "🌿 Improves digestion",
            "💨 Relieves cold and cough",
            "🦷 Improves oral health",
            "🧠 Reduces stress"
        ]
    },
    12: {
        "name": "Neem",
        "uses": [
            "🩸 Blood purification",
            "🌿 Treats skin diseases",
            "🦷 Dental care",
            "💪 Boosts immunity"
        ]
    },
    25: {
        "name": "Tulsi",
        "uses": [
            "🤧 Treats cold and fever",
            "💪 Improves immunity",
            "🫁 Improves respiratory health"
        ]
    }
}

# ---------------- CSS ----------------
st.markdown("""
<style>
body{
    background:linear-gradient(135deg,#e0f7fa,#f1f8e9);
}
.header{
    background:linear-gradient(135deg,#7ddc9c,#5bbf7a);
    padding:25px;
    text-align:center;
    font-size:30px;
    color:#0f3d2e;
    border-radius:0 0 40px 40px;
}
.box{
    background:#ffffff;
    padding:40px;
    border-radius:30px;
    box-shadow:0 10px 25px rgba(0,0,0,0.2);
    text-align:center;
}
.result{
    margin-top:50px;
    background:#f1fff8;
    padding:40px;
    border-radius:30px;
}
.card{
    background:#ffffff;
    padding:20px;
    border-radius:25px;
    box-shadow:0 10px 25px rgba(0,0,0,0.2);
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='header'>🌿 Herbal Plant Identification System</div>", unsafe_allow_html=True)
st.write("")

# ---------------- UPLOAD SECTION ----------------
st.markdown("<div class='box'>", unsafe_allow_html=True)
st.markdown("## Upload Plant Image 🌸")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
identify = st.button("✨ Identify Plant")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RESULT SECTION ----------------
if identify and uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    st.markdown("<div class='result'>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, use_column_width=True)

    with col2:
        if class_id in PLANT_INFO:
            plant = PLANT_INFO[class_id]
            st.success(f"🌱 Identified Plant: {plant['name']}")
            st.write(f"**Confidence:** {confidence:.2f}")
            st.markdown("### 🌿 Medicinal Uses")
            for use in plant["uses"]:
                st.markdown(f"- {use}")
        else:
            st.error("❌ This is NOT a Herbal / Medicinal Plant")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- COMMON PLANTS ----------------
st.write("")
st.markdown("## 🌼 Common Medicinal Plants")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("<div class='card'><h4>Tulsi</h4><p>Boosts immunity and treats fever.</p></div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'><h4>Neem</h4><p>Used for skin care and blood purification.</p></div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='card'><h4>Turmeric</h4><p>Natural anti-inflammatory plant.</p></div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='card'><h4>Aloe Vera</h4><p>Used for skin and digestion.</p></div>", unsafe_allow_html=True)
