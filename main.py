import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import ollama

# Load the trained model safely
model = tf.keras.models.load_model('plant_disease_prediction_model.h5', compile=False)

# Check model input shape
expected_input_shape = model.input_shape[1:3]  # Extract height and width
print(f"Model expects input shape: {expected_input_shape}")

# Define class labels
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure the image has 3 channels
    image = image.resize(expected_input_shape)  # Resize to match model input shape
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Streamlit UI
st.title("🌿 Plant Disease Detection with AI + LLM")

# Upload Image
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image & predict
    processed_image = preprocess_image(image)
    
    # Debug: Print shape before passing to model
    print(f"Processed image shape: {processed_image.shape}")

    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]

    st.write(f"**Prediction:** {predicted_class}")

    