import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Herbal Plant Identification", layout="wide")

# ---------------- CUSTOM HTML-LIKE CSS ----------------
st.markdown("""
<style>

/* REMOVE STREAMLIT WHITE BOXES */
.block-container {
    padding-top: 0rem;
}
.stAlert, .stButton, .stFileUploader {
    background: transparent !important;
}

/* BODY */
.stApp {
    min-height:100vh;
    background:
        linear-gradient(rgba(224,247,250,0.9), rgba(241,248,233,0.9)),
        url("https://images.unsplash.com/photo-1466692476868-aef1dfb1e735")
        center/cover fixed;
    font-family:'Segoe UI', Tahoma, sans-serif;
}

/* HEADER */
.header {
    background: linear-gradient(135deg,#7ddc9c,#5bbf7a);
    color:#0f3d2e;
    padding:24px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
    border-bottom-left-radius:40px;
    border-bottom-right-radius:40px;
    box-shadow:0 8px 20px rgba(0,0,0,0.25);
}

/* UPLOAD BOX */
.upload-box {
    background:linear-gradient(135deg,#fff1f8,#e8fff3);
    border-radius:40px;
    padding:45px;
    text-align:center;
    margin:40px auto;
    border:4px dashed #a5e6c6;
    box-shadow:0 20px 35px rgba(0,0,0,0.25);
}

.upload-box h2 {
    color:#2e7d32;
}

.upload-box p {
    color:#4a7c6a;
}

/* BUTTON */
.stButton>button {
    background:linear-gradient(135deg,#ffb6d5,#ff8fb8);
    border:none;
    padding:15px 38px;
    color:white;
    font-size:16px;
    border-radius:50px;
    box-shadow:0 10px 20px rgba(255,143,184,0.6);
    transition:0.35s;
}
.stButton>button:hover {
    transform:scale(1.08);
}

/* RESULT */
.result-box {
    background:linear-gradient(135deg,#e6fff6,#f0fff9);
    padding:40px;
    border-radius:45px;
    box-shadow:0 25px 45px rgba(0,0,0,0.3);
    border:5px solid #d1f7e5;
}

.result-box h3 {
    color:#1b5e20;
}
.result-box p, li {
    color:#3f6f60;
    line-height:1.8;
}

/* COMMON PLANTS */
.section-title {
    text-align:center;
    font-size:32px;
    color:white;
    margin:70px 0 30px;
    text-shadow:0 6px 18px rgba(0,0,0,0.6);
}

.plant-card {
    background:linear-gradient(135deg,#fff6fb,#f1fff8);
    border-radius:45px;
    text-align:center;
    padding:20px;
    box-shadow:0 20px 35px rgba(0,0,0,0.25);
    transition:0.35s;
}
.plant-card:hover {
    transform:translateY(-10px);
}

.plant-card h4 {
    color:#2e7d32;
}

/* FOOTER */
.footer {
    margin-top:80px;
    background:linear-gradient(135deg,#7ddc9c,#5bbf7a);
    color:#0f3d2e;
    text-align:center;
    padding:18px;
    border-top-left-radius:40px;
    border-top-right-radius:40px;
    box-shadow:0 -8px 20px rgba(0,0,0,0.3);
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

CLASS_NAMES = ["Mint", "Neem", "Tulasi", "Aloevera"]

DESCRIPTIONS = {
    "Adas": "Adas is an aromatic herb commonly used in traditional medicine. It supports digestion and relieves bloating.",
    "Aloevera": "A succulent medicinal plant known for its soothing gel. Widely used for skin care and digestive health.",
    "Amla": "A nutrient-rich fruit high in vitamin C. Commonly used to boost immunity and improve hair health.",
    "Amruta_Balli": "A climbing shrub used in Ayurveda. Known for immunity boosting and fever management properties.",
    "Andong Merah": "A traditional medicinal plant used in herbal remedies. Known for anti-inflammatory and wound healing effects.",
    "Arali": "An ornamental plant with medicinal value. Traditionally used for general wellness.",
    "Ashoka": "A sacred medicinal tree in Ayurveda. Commonly used for gynecological and menstrual health.",
    "Ashwagandha": "A powerful adaptogenic herb. Used to reduce stress and improve strength and vitality.",
    "Avacado": "A nutritious fruit rich in healthy fats. Supports heart health and overall nutrition.",
    "Bamboo": "A fast-growing plant used in herbal medicine. Known for bone strengthening and anti-inflammatory properties.",
    "Basale": "A leafy vegetable with medicinal benefits. Helps digestion and reduces inflammation.",
    "Belimbing Wulu": "A medicinal fruit plant rich in antioxidants. Commonly used for digestive health.",
    "Beluntas": "A shrub used in traditional medicine. Helps control body odor and supports digestion.",
    "Betadin": "A medicinal plant with antiseptic properties. Traditionally used for wound care.",
    "Betel": "A widely used medicinal leaf in Asia. Supports oral health and digestion.",
    "Betel_Nut": "A traditional chewing nut. Used as a digestive stimulant in herbal practices.",
    "Brahmi": "A well-known brain tonic herb. Enhances memory and cognitive function.",
    "Castor": "A medicinal plant known for castor oil. Used as a laxative and for joint pain relief.",
    "Cincau Perdu": "A cooling medicinal plant. Helps in digestion and body cooling.",
    "Curry_Leaf": "An aromatic leaf used in cooking and medicine. Helps regulate digestion and blood sugar.",
    "Daun Afrika": "A medicinal plant used in folk medicine. Known for immunity support and anti-cancer properties.",
    "Daun Cabe Jawa": "A traditional herbal plant. Used to improve digestion and reduce inflammation.",
    "Daun Cocor Bebek": "A succulent medicinal plant. Known for wound healing and anti-inflammatory effects.",
    "Daun Kumis Kucing": "A kidney-supporting medicinal plant. Acts as a natural diuretic.",
    "Daun Mangkokan": "A leafy medicinal plant. Commonly used for hair growth and skin care.",
    "Daun Suji": "A natural coloring plant with medicinal value. Supports digestion.",
    "Daun Ungu": "A traditional herbal plant. Used to treat hemorrhoids and inflammation.",
    "Dewa Ndaru": "A rare medicinal plant in folk medicine. Used for general health improvement.",
    "Doddapatre": "An aromatic medicinal herb. Commonly used to treat cold and digestion issues.",
    "Ekka": "A traditional medicinal shrub. Used for pain relief and inflammation.",
    "Gandarusa": "A medicinal plant used for joint pain. Known for anti-inflammatory effects.",
    "Ganike": "A lesser-known herbal plant. Used in traditional wellness treatments.",
    "Garut": "A tuber plant used in herbal food remedies. Helps digestion and boosts energy.",
    "Gauva": "A fruit-bearing medicinal tree. Used to control diarrhea and boost immunity.",
    "Geranium": "An aromatic medicinal plant. Used in skin care and aromatherapy.",
    "Henna": "A natural dye plant with cooling properties. Supports hair and skin health.",
    "Hibiscus": "A flowering medicinal plant. Promotes hair growth and heart health.",
    "Honge": "A medicinal tree used in Ayurveda. Known for treating skin diseases.",
    "Honje": "A traditional herbal plant. Supports digestion and antioxidant activity.",
    "Iler": "A medicinal ornamental plant. Used for wound healing and inflammation.",
    "Insulin": "A medicinal plant used for diabetes management. Helps regulate blood sugar levels.",
    "Jahe": "Commonly known as ginger. Widely used for digestion and nausea relief.",
    "Jasmine": "A fragrant flowering plant. Used for relaxation and aromatherapy.",
    "Jeruk Nipis": "A citrus medicinal fruit. Rich in vitamin C and detoxifying properties.",
    "Kapulaga": "Aromatic spice plant. Improves digestion and freshens breath.",
    "Kayu Putih": "A medicinal tree producing eucalyptus oil. Used for cold and muscle pain relief.",
    "Kecibling": "A medicinal herb supporting urinary health. Acts as a diuretic.",
    "Kemangi": "An aromatic herbal leaf. Supports digestion and fights bacteria.",
    "Kembang Sepatu": "A flowering medicinal plant. Used for hair care and blood pressure control.",
    "Kenanga": "A fragrant medicinal flower. Used in stress relief and aromatherapy.",
    "Kunyit": "Also known as turmeric. Famous for anti-inflammatory and healing properties.",
    "Lampes": "A traditional medicinal plant. Used in folk remedies for wellness.",
    "Legundi": "A herbal plant used for respiratory health. Reduces inflammation.",
    "Lemon": "A citrus medicinal fruit. Used for detoxification and immunity.",
    "Lemon_grass": "A fragrant medicinal grass. Used for digestion and stress relief.",
    "Lidah Buaya": "Another name for aloe vera. Used for skin healing and digestion.",
    "Mahkota Dewa": "A powerful medicinal plant. Known for blood purification and anti-cancer properties.",
    "Mango": "A tropical fruit tree with medicinal value. Supports digestion and immunity.",
    "Melati": "A fragrant medicinal flower. Used in skin care and relaxation.",
    "Meniran": "A traditional herbal plant. Supports liver health and immunity.",
    "Mint": "A cooling aromatic herb. Used for digestion and cold relief.",
    "Murbey": "A medicinal fruit plant. Helps regulate blood sugar and acts as antioxidant.",
    "Nagadali": "A folk medicinal plant. Used for general wellness.",
    "Neem": "A powerful medicinal tree. Known for antibacterial and blood purifying properties.",
    "Nilam": "Aromatic medicinal plant. Used in perfumes and skin care.",
    "Nithyapushpa": "A traditional medicinal flower. Used for general health.",
    "Nooni": "Also known as noni plant. Used to boost immunity and digestion.",
    "Pacing Petul": "A medicinal herb. Used for pain relief and inflammation.",
    "Pandan": "An aromatic medicinal plant. Used in digestion and relaxation.",
    "Pappaya": "Papaya plant used medicinally. Supports digestion and gut health.",
    "Patah Tulang": "A medicinal shrub. Known for bone healing properties.",
    "Pecut Kuda": "A traditional medicinal plant. Used for inflammation and pain relief.",
    "Pepper": "A spice plant with medicinal benefits. Improves digestion and treats cold.",
    "Pomegranate": "A medicinal fruit rich in antioxidants. Supports heart health.",
    "Raktachandini": "A medicinal tree. Used for skin diseases and blood purification.",
    "Rose": "A fragrant flowering plant. Used in skin care and aromatherapy.",
    "Saga Manis": "A medicinal climber plant. Used for respiratory and cough relief.",
    "Sapota": "A fruit tree with medicinal value. Supports digestion and energy.",
    "Secang": "A medicinal wood plant. Used for blood purification and antioxidants.",
    "Sereh": "Another name for lemongrass. Used for digestion and stress relief.",
    "Sirih": "A medicinal leaf plant. Used for oral health and antibacterial purposes.",
    "Srikaya": "A fruit-bearing medicinal plant. Supports digestion and antioxidants.",
    "Tin": "Also known as fig plant. Rich in fiber and digestive benefits.",
    "Tulasi": "A sacred medicinal plant in India. Used for immunity and respiratory health.",
    "Wood_sorel": "A medicinal leafy plant. Used as cooling agent and digestive aid.",
    "Zigzag": "An ornamental medicinal plant. Used in traditional herbal remedies."

}

USES = {
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

# ---------------- UPLOAD ----------------
st.markdown("""
<div class='upload-box'>
<h2>Upload Plant Image 🌸</h2>
<p>Select a leaf or plant image to identify its medicinal uses</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])
identify = st.button("Identify Plant ✨")

# ---------------- RESULT ----------------
if identify and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = image.resize((224,224))
    img = np.expand_dims(np.array(img)/255.0, axis=0)

    preds = model.predict(img)
    plant = CLASS_NAMES[int(np.argmax(preds))]

    col1, col2 = st.columns([1,2])

    with col1:
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{plant}</h3>", unsafe_allow_html=True)
        st.write(DESCRIPTIONS.get(plant))
        st.markdown("**Medicinal Uses:**")
        for u in USES.get(plant, []):
            st.write("🌿", u)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- COMMON PLANTS ----------------
st.markdown("<div class='section-title'>🌼 Common Medicinal Plants 🌼</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, plant in zip([c1,c2,c3,c4], CLASS_NAMES):
    with col:
        st.markdown(
            f"<div class='plant-card'><h4>{plant}</h4></div>",
            unsafe_allow_html=True
        )

