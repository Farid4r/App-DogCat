import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model_cat_dog.keras")

st.title("Cat & Dog Classifier")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

THRESHOLD = 70.0

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)[0][0]

    if result < 0.5:
        prediction = "Cat"
        confidence = (1 - result) * 100
    else:
        prediction = "Dog"
        confidence = result * 100

    confidence = round(confidence, 2)

    if confidence < THRESHOLD:
        st.warning("The image is neither a cat nor a dog.")
    else:
        st.success(f"Prediction: {prediction}")
        st.write(f"Confidence: {confidence}%")
