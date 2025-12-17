import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Herbal Plant Identification", layout="wide")

MODEL_URL = "https://drive.google.com/file/d/1a-_536wX34s8nakc84eI6TPatxmidKva/view"
MODEL_PATH = "herbal_model.h5"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first time only)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background:
        url("https://images.unsplash.com/photo-1466692476868-aef1dfb1e735") center/cover fixed,
        linear-gradient(135deg,#e0f7fa,#f1f8e9);
}

header {
    background:linear-gradient(135deg,#7ddc9c,#5bbf7a);
    padding:25px;
    text-align:center;
    font-size:32px;
    color:#0f3d2e;
    border-radius:0 0 40px 40px;
    box-shadow:0 8px 20px rgba(0,0,0,0.25);
    margin-bottom:40px;
}

.upload-box {
    background:linear-gradient(135deg,#fff1f8,#e8fff3);
    padding:45px;
    border-radius:40px;
    text-align:center;
    box-shadow:0 20px 35px rgba(0,0,0,0.25);
    border:4px dashed #a5e6c6;
}

.result-box {
    margin-top:60px;
    background:linear-gradient(135deg,#e6fff6,#f0fff9);
    padding:40px;
    border-radius:45px;
    box-shadow:0 25px 45px rgba(0,0,0,0.3);
    border:5px solid #d1f7e5;
}

.plant-card {
    background:linear-gradient(135deg,#fff6fb,#f1fff8);
    border-radius:35px;
    box-shadow:0 20px 35px rgba(0,0,0,0.25);
    padding:20px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<header>🌿 Herbal Plant Identification System</header>", unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ----------------
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
st.markdown("## Upload Plant Image 🌸")
st.markdown("Select a leaf or plant image to identify its medicinal uses")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if uploaded_file:
    col1, col2 = st.columns([1, 2])

    image = Image.open(uploaded_file).convert("RGB")

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # Example mapping (EDIT THIS)
    plant_names = {
        60: ("Mint (Pudina)", [
            "🌿 Improves digestion",
            "💨 Relieves cold",
            "🧠 Improves focus",
            "🦷 Oral health"
        ])
    }

    plant, uses = plant_names.get(class_id, ("Unknown Plant", []))

    with col1:
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"### 🌱 {plant}")
        st.markdown(f"**Confidence:** {confidence:.2f}")
        st.markdown("**Medicinal Uses:**")
        for u in uses:
            st.markdown(f"- {u}")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- COMMON PLANTS ----------------
st.markdown("## 🌼 Common Medicinal Plants 🌼")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("<div class='plant-card'><h4>Tulsi</h4><p>Boosts immunity & treats cold.</p></div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='plant-card'><h4>Turmeric</h4><p>Anti-inflammatory & healer.</p></div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='plant-card'><h4>Ginger</h4><p>Aids digestion & nausea.</p></div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='plant-card'><h4>Aloe Vera</h4><p>Skin & digestive health.</p></div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>🌸 2025 Herbal AI Project | Aesthetic UI 🌱</center>
""", unsafe_allow_html=True)
