import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# Page config
st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="centered")

# Title
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish, and I’ll tell you its species.")

# Load model and class names
@st.cache_resource
def load_model_and_labels():
    model = load_model("mobilenetv2_model.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_labels = load_model_and_labels()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred_prob = model.predict(img_array)[0]
    pred_idx = np.argmax(pred_prob)
    pred_class = class_labels[pred_idx]
    confidence = pred_prob[pred_idx] * 100

    # Show results
    st.markdown(f"### 🎯 Prediction: **{pred_class}**")
    st.markdown(f"### 📊 Confidence: **{confidence:.2f}%**")

    # Show probability chart
    st.bar_chart({label: prob for label, prob in zip(class_labels, pred_prob)})
