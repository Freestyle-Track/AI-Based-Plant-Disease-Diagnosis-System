import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from langchain.llms import Ollama  #Ollama (Mistral)
import requests
import io
import ollama
# Load the trained model
model = tf.keras.models.load_model('plant_disease_prediction_model.h5')

# Define class labels
class_labels =['Apple___Apple_scab',
                'Apple___Black_rot',
                'Apple___Cedar_apple_rust',
                'Apple___healthy',
                'Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy',
                'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot',
                'Peach___healthy',
                'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy',
                'Raspberry___healthy',
                'Soybean___healthy',
                'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch',
                'Strawberry___healthy',
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
                ]

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to get disease information from LLM
def get_disease_info(disease_name):
    prompt = f"Give a detailed explanation about {disease_name}. How does it occur, and what are its treatments?"

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])

    return response['message']['content']

# Streamlit UI
st.title("🌿 Plant Disease Detection with AI + LLM")

# Upload Image
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image & predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]

    st.write(f"**Prediction:** {predicted_class}")

    # Get additional info using LLM
    with st.spinner("Fetching disease details..."):
        disease_info = get_disease_info(predicted_class)

    st.write("### More Information:")
    st.write(disease_info)
