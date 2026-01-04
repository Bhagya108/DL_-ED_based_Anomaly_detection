import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(
    page_title="🔥 Fire Detection System",
    layout="centered"
)

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
MODEL_PATH = "fire_detection_lightweight_cnn.tflite"

# =========================
# LOAD TFLITE MODEL
# =========================
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# UI
# =========================
st.title("🔥 Fire Detection System")
st.write("Upload an image to detect **Fire / No Fire**")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION FUNCTION
# =========================
def predict(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )[0][0]

    label = "🔥 FIRE" if prediction > 0.5 else "✅ NO FIRE"
    return label, float(prediction)

# =========================
# DISPLAY RESULT
# =========================
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        label, confidence = predict(image)

    st.subheader("Prediction")
    st.markdown(f"### {label}")
    st.write(f"**Confidence:** `{confide
