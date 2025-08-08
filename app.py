import streamlit as st
import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.title("🐟 Fish Species Classifier (MobileNetV2)")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

@st.cache_resource
def load_keras_model():
    try:
        # Try to load as a full saved model
        model = tf.keras.models.load_model("mobilenetv2_model.keras")
        st.write("✅ Loaded full model.")
    except Exception:
        # If that fails, assume weights-only and rebuild architecture
        st.write("ℹ️ Loading weights-only model...")
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(300, 300, 3),
            pooling='avg'
        )
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        model.load_weights("mobilenetv2_model.keras")
        st.write("✅ Weights loaded into rebuilt architecture.")
    return model

model = load_keras_model()

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((300, 300))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Classifying..."):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    st.success(f"**Predicted:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")


# streamlit run app.py 
