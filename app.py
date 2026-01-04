import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NO FIRE ✅", "FIRE 🔥"]

st.set_page_config(
    page_title="🔥 Fire Detection System",
    layout="centered"
)

# =========================
# LOAD TFLITE MODEL
# =========================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(
        model_path="fire_detection_lightweight_cnn.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype(np.float32)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

    label = CLASS_NAMES[int(prediction > 0.5)]
    confidence = float(prediction)

    return label, confidence

# =========================
# UI
# =========================
st.title("🔥 Fire Detection System")
st.write("Upload an image to detect **Fire / No Fire** using a deep learning model.")

uploaded_file = st.file_uploader(
    "Upload an image
