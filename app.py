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

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* REMOVE TOP GAP */
.block-container {
    padding-top: 0rem;
}

/* BACKGROUND */
.stApp {
    background: linear-gradient(rgba(245,255,250,0.85), rgba(245,255,250,0.85)),
                url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6')
                center/cover fixed;
}

/* HEADER */
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

/* GLASS CARD */
.card, .result-box, .uses-box, .common {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(15px);
    border-radius: 30px;
    padding: 25px;
    box-shadow: 0 15px 30px rgba(0,0,0,0.15);
}

/* UPLOAD TITLE */
.upload-title {
    font-size: 22px;
    font-weight: bold;
    color: #1e4f3b;
    margin-bottom: 10px;
}

/* FILE UPLOADER TEXT */
label, .stFileUploader label {
    color: #1e4f3b !important;
    font-size: 18px !important;
    font-weight: bold !important;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg,#81c784,#2e7d32);
    color: white;
    font-weight: bold;
    border-radius: 50px;
    padding: 10px 25px;
}

/* SECTION TITLE */
.section-title {
    font-size: 26px;
    font-weight: bold;
    color: #1e4f3b;
    margin-top: 30px;
}

/* COMMON PLANTS */
.common {
    text-align: center;
    color: #2e7d32;
    font-weight: bold;
    border: 2px solid #a5e6c6;
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
    "Adas": ["Digestive aid", "Relieves bloating", "Improves appetite"],
    "Aloevera": ["Skin care", "Burn healing", "Digestive health"],
    "Amla": ["Vitamin C rich", "Boosts immunity", "Hair health"],
    "Amruta_Balli": ["Immunity booster", "Fever management", "Detoxification"],
    "Andong Merah": ["Anti-inflammatory", "Wound healing"],
    "Arali": ["Used in traditional remedies", "Supports wellness"],
    "Ashoka": ["Gynecological health", "Menstrual regulation"],
    "Ashwagandha": ["Stress relief", "Strength & vitality"],
    "Avacado": ["Heart health", "Rich in healthy fats"],
    "Bamboo": ["Anti-inflammatory", "Bone health"],
    "Basale": ["Improves digestion", "Anti-inflammatory"],
    "Belimbing Wulu": ["Antioxidant", "Digestive aid"],
    "Beluntas": ["Body odor control", "Digestive health"],
    "Betadin": ["Antiseptic plant", "Wound care"],
    "Betel": ["Oral health", "Digestive stimulant"],
    "Betel_Nut": ["Digestive stimulant", "Traditional chewing plant"],
    "Brahmi": ["Memory enhancement", "Brain tonic"],
    "Castor": ["Laxative", "Joint pain relief"],
    "Cincau Perdu": ["Cooling agent", "Digestive relief"],
    "Curry_Leaf": ["Controls diabetes", "Improves digestion"],
    "Daun Afrika": ["Anti-cancer properties", "Immunity support"],
    "Daun Cabe Jawa": ["Digestive aid", "Anti-inflammatory"],
    "Daun Cocor Bebek": ["Wound healing", "Anti-inflammatory"],
    "Daun Kumis Kucing": ["Kidney health", "Diuretic"],
    "Daun Mangkokan": ["Hair growth", "Skin care"],
    "Daun Suji": ["Natural coloring", "Digestive aid"],
    "Daun Ungu": ["Hemorrhoid relief", "Anti-inflammatory"],
    "Dewa Ndaru": ["Traditional healing plant", "General wellness"],
    "Doddapatre": ["Cold relief", "Digestive aid"],
    "Ekka": ["Pain relief", "Anti-inflammatory"],
    "Gandarusa": ["Rheumatic pain relief", "Anti-inflammatory"],
    "Ganike": ["Traditional herbal remedy", "General wellness"],
    "Garut": ["Digestive health", "Energy booster"],
    "Gauva": ["Controls diarrhea", "Rich in antioxidants"],
    "Geranium": ["Aromatherapy", "Skin care"],
    "Henna": ["Cooling effect", "Hair & skin health"],
    "Hibiscus": ["Hair growth", "Blood pressure regulation"],
    "Honge": ["Skin diseases", "Anti-inflammatory"],
    "Honje": ["Digestive aid", "Anti-oxidant"],
    "Iler": ["Anti-inflammatory", "Wound healing"],
    "Insulin": ["Blood sugar control", "Diabetes management"],
    "Jahe": ["Digestive aid", "Reduces nausea"],
    "Jasmine": ["Stress relief", "Aromatherapy"],
    "Jeruk Nipis": ["Vitamin C source", "Detoxification"],
    "Kapulaga": ["Improves digestion", "Freshens breath"],
    "Kayu Putih": ["Cold relief", "Muscle pain relief"],
    "Kecibling": ["Urinary health", "Diuretic"],
    "Kemangi": ["Digestive aid", "Anti-bacterial"],
    "Kembang Sepatu": ["Hair care", "Blood pressure control"],
    "Kenanga": ["Stress relief", "Aromatherapy"],
    "Kunyit": ["Anti-inflammatory", "Wound healing"],
    "Lampes": ["Traditional herbal medicine", "General wellness"],
    "Legundi": ["Respiratory health", "Anti-inflammatory"],
    "Lemon": ["Vitamin C", "Detoxification"],
    "Lemon_grass": ["Stress reduction", "Digestive aid"],
    "Lidah Buaya": ["Skin care", "Digestive health"],
    "Mahkota Dewa": ["Anti-cancer properties", "Blood purification"],
    "Mango": ["Digestive enzymes", "Immunity booster"],
    "Melati": ["Stress relief", "Skin care"],
    "Meniran": ["Liver protection", "Immunity booster"],
    "Mint": ["Digestive aid", "Cold relief"],
    "Murbey": ["Blood sugar regulation", "Antioxidant"],
    "Nagadali": ["Traditional remedy", "General wellness"],
    "Neem": ["Blood purification", "Skin diseases"],
    "Nilam": ["Aromatherapy", "Skin care"],
    "Nithyapushpa": ["Traditional medicine", "General wellness"],
    "Nooni": ["Digestive aid", "Immunity booster"],
    "Pacing Petul": ["Anti-inflammatory", "Pain relief"],
    "Pandan": ["Digestive aid", "Aromatherapy"],
    "Pappaya": ["Digestive enzymes", "Gut health"],
    "Patah Tulang": ["Bone healing", "Anti-inflammatory"],
    "Pecut Kuda": ["Anti-inflammatory", "Traditional medicine"],
    "Pepper": ["Improves digestion", "Cold relief"],
    "Pomegranate": ["Antioxidant rich", "Heart health"],
    "Raktachandini": ["Skin diseases", "Blood purifier"],
    "Rose": ["Skin hydration", "Aromatherapy"],
    "Saga Manis": ["Respiratory health", "Cough relief"],
    "Sapota": ["Energy booster", "Digestive health"],
    "Secang": ["Blood purification", "Anti-oxidant"],
    "Sereh": ["Digestive aid", "Stress relief"],
    "Sirih": ["Oral health", "Anti-bacterial"],
    "Srikaya": ["Digestive aid", "Antioxidant"],
    "Tin": ["Digestive health", "Rich in fiber"],
    "Tulasi": ["Cold & cough", "Immunity booster"],
    "Wood_sorel": ["Cooling agent", "Digestive aid"],
    "Zigzag": ["Traditional ornamental medicinal plant"]
}

CONFIDENCE_THRESHOLD = 0.75

# ---------------- UPLOAD SECTION ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='upload-title'>📤 Upload Plant Image</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Choose an image file",
    type=["jpg", "jpeg", "png"]
)

identify = st.button("✨ Identify Plant")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if identify and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    plant_name = CLASS_NAMES[class_id]

    col1, col2 = st.columns([1, 2])

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
for col, name in zip([c1, c2, c3, c4], ["Tulasi", "Neem", "Mint", "Aloevera"]):
    with col:
        st.markdown(f"<div class='common'><h4>{name}</h4></div>", unsafe_allow_html=True)
