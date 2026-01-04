import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
CLASS_NAMES = ["fire", "non_fire"]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fire_detection_lightweight_cnn.h5")
    return model

model = load_model()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="🔥 Fire Detection System",
    layout="centered"
)

st.title("🔥 Fire Detection using CNN")
st.write("Upload an image to detect **Fire / No Fire**")

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]

    label = "FIRE 🔥" if prediction > 0.5 else "NO FIRE ✅"
    confidence = float(prediction)

    return label, confidence

# =========================
# DISPLAY RESULT
# =========================
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        label, confidence = predict_image(image)

    st.subheader("Prediction Result")
    st.write(f"### {label}")
    st.write(f"**Confidence:** `{confidence:.2f}`")

    if "FIRE" in label:
        st.error("⚠️ Fire detected! Take action immediately.")
    else:
        st.success("✅ No fire detected. Environment looks safe.")
