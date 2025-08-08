import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------------
# Load the trained model
# ------------------------
MODEL_PATH = "mobilenetv2_model_['Catla', 'CommonCarp', 'Mori', 'Rohu', 'SilverCarp'].h5"
model = load_model(MODEL_PATH)

# Class labels (must match training order)
class_labels = ['Catla', 'CommonCarp', 'Mori', 'Rohu', 'SilverCarp']

# ------------------------
# Prediction function
# ------------------------
def predict_single_image(img, model, class_labels):
    img = img.resize((300, 300))  # Resize to match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_prob = model.predict(img_array)[0]
    pred_idx = np.argmax(pred_prob)

    return class_labels[pred_idx], pred_prob[pred_idx], pred_prob

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟", layout="centered")

st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish, and the model will predict its species.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prediction
    with st.spinner("Predicting..."):
        pred_class, confidence, all_probs = predict_single_image(img, model, class_labels)

    # Display results
    st.markdown(f"### 🏆 Predicted Class: **{pred_class}**")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

    # Show all class probabilities as a bar chart
    st.subheader("Class Probabilities")
    prob_dict = {class_labels[i]: float(all_probs[i]) for i in range(len(class_labels))}
    st.bar_chart(prob_dict)
else:
    st.info("Please upload an image to get a prediction.")
