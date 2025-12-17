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
    "Ashoka","Ashwagandha","Avacado","Bamboo","
