import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model_cat_dog.keras")

st.set_page_config(page_title="Klasifikasi Kucing & Anjing")

st.title("🐱🐶 Klasifikasi Gambar Kucing dan Anjing")
st.write("Upload gambar kucing atau anjing untuk diprediksi oleh model.")

uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)[0][0]

    if result < 0.5:
        prediction = "Kucing"
        confidence = (1 - result) * 100
    else:
        prediction = "Anjing"
        confidence = result * 100

    confidence = round(confidence, 2)

    st.success(f"Hasil Prediksi: {prediction}")
    st.write(f"Tingkat Keyakinan Model: {confidence}%")
