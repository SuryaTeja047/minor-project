import streamlit as st
import numpy as np
import os
import json
from PIL import Image
import tensorflow as tf
from classify import classify_image

# Load class labels (ensure this matches your trained model's labels)
class_labels = [
    'Aloevera', 'Amla', 'Amruta Balli', 'Arali', 'Ashoka', 'Ashwagandha',
    'Avocado', 'Bamboo', 'Basale', 'Betel', 'Betel Nut', 'Brahmi', 'Castor',
    
    'Curry Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Guava', 'Geranium',
    'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon', 'Lemon Grass',
    'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Papaya',
    'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi',
    'Wood Sorel'
]

# Load plant information from JSON file
working_dir = os.path.dirname(os.path.abspath(__file__))
plant_info_path = os.path.join(working_dir, 'plant_info.json')

# Check if plant_info.json exists and load it
if os.path.exists(plant_info_path):
    with open(plant_info_path, 'r') as f:
        plant_info = json.load(f)
else:
    st.error("Plant information file not found. Please ensure 'plant_info.json' is in the directory.")
    plant_info = {}

# Load remedies for diseases
remedies_path = os.path.join(working_dir, 'remedies.json')

if os.path.exists(remedies_path):
    with open(remedies_path, 'r') as f:
        remedies = json.load(f)
else:
    st.error("Remedies information file not found. Please ensure 'remedies.json' is in the directory.")
    remedies = {}

# Paths and model loading for the Plant Disease Classifier
model_path = f"{working_dir}/models/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to predict the class of an image for the Disease Classifier
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Medicinal Plant Identification", "Plant Disease Classifier"])

# Home Page
if app_mode == "Home":
    st.title("Welcome to Plant Sense")
    st.image("images/homepage.jpg", use_column_width=True)

    st.markdown("""
    **Plant Sense** offers a powerful solution for identifying medicinal plants and detecting plant diseases using advanced machine learning techniques.

    ### How It Works
    1. **Upload Image:** Go to the **Plant Identification** or **Plant Disease Classifier** page to upload a plant image.
    2. **Analysis:** Our model will analyze the image to determine the plant species and its potential uses or identify diseases.
    3. **Results:** View detailed results and learn about the plant's medicinal properties or disease information.

    ### Why Choose Plant Sense?
    - **Precision:** Our CNN-based models ensure high accuracy in plant identification and disease classification.
    - **Ease of Use:** Intuitive interface designed for a seamless experience.
    - **Rapid Results:** Get instant feedback and recommendations.

    ### Explore More
    Visit the **Plant Identification** or **Plant Disease Classifier** page to start using our system!
    """)
# About Page
elif app_mode == "About":
    st.title("About This Project")
    st.markdown("""
    ### Dataset Overview
    This project includes around 90,000 RGB images of medicinal plants, categorized into 40 species. Enhanced through offline augmentation techniques, the dataset is divided into training and validation sets (80/20) and includes a separate directory for test images.

    ### Technologies Used
    - **Convolutional Neural Networks (CNNs):** For advanced image classification.
    - **TensorFlow:** Framework for model training and evaluation.
    - **Streamlit:** For building an interactive and user-friendly interface.
    - **NumPy:** For numerical operations and data handling.
    - **Pandas:** For preprocessing and managing data.
    - **Matplotlib:** For visualizing model performance and results.

    This project aims to support accurate and efficient medicinal plant identification and disease detection using state-of-the-art technology.
    """)

elif app_mode == "Medicinal Plant Identification":
    st.title("Medicinal Plant Identification")
    st.markdown("""
    Upload an image of a medicinal plant to get its identification and learn about its uses.

    ### Instructions
    1. Click **Browse files** to upload a plant image.
    2. Click **Show Image** to preview the uploaded image.
    3. Click **Predict** to receive the plant's identification and information.
    """)

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    # Display the uploaded image
    # Display the uploaded image with smaller dimensions
    if st.button("Show Image", key="show_image_identification"):
        if test_image is not None:
            st.image(test_image, caption="Uploaded Image", width=300)  # Set the width to 300px
        else:
            st.warning("Please upload an image first.")


    # Prediction button
    if st.button("Predict", key="predict_identification"):
        if test_image is not None:
            st.write("Analyzing the image...")

            # Save the uploaded image to a temporary file
            temp_image_path = "temp_image.jpg"
            with open(temp_image_path, "wb") as temp_file:
                temp_file.write(test_image.getbuffer())

            # Get classification results
            predictions = classify_image(temp_image_path)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]

            # Display result
            # Display result
            st.success(f"Prediction: {predicted_class}")

            # Retrieve and display plant information
            plant_data = plant_info.get(predicted_class)

            if plant_data:
                st.write("### Plant Description and Uses")
                st.write(f"**Description:** {plant_data.get('description', 'No description available.')}")
                st.write(f"**Uses:** {plant_data.get('uses', 'No uses available.')}")
            else:
                st.write("Description and uses information not available.")

        else:
            st.warning("Please upload an image first.")
# Plant Disease Classifier Page
elif app_mode == "Plant Disease Classifier":
    st.title('Plant Disease Classifier')
    st.markdown("""
    Upload an image of a plant to detect its disease.

    ### Instructions
    1. Click **Browse files** to upload a plant image.
    2. Click **Show Image** to preview the uploaded image.
    3. Click **Predict** to identify the disease and get remedy information.
    """)

    uploaded_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    # Display the uploaded image
    # Display the uploaded image with smaller dimensions
    if st.button("Show Image", key="show_image_identification"):
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", width=300)  # Set the width to 300px
        else:
            st.warning("Please upload an image first.")


    # Prediction button
    if st.button("Predict", key="predict_disease"):
        if uploaded_image is not None:
            st.write("Analyzing the image...")

            # Save the uploaded image to a temporary file
            temp_image_path = "temp_image.jpg"
            with open(temp_image_path, "wb") as temp_file:
                temp_file.write(uploaded_image.getbuffer())

            # Get classification results
            predicted_class = predict_image_class(model, temp_image_path, class_indices)
            st.success(f"Prediction: {predicted_class}")

            # Check if plant is healthy
            if predicted_class == "Healthy":
                st.write("The plant appears to be healthy with no visible disease symptoms.")
            else:
                # Display remedy for the predicted disease
                remedy = remedies.get(predicted_class).get('remedy')
                if remedy:
                    st.write(f"### Remedy for {predicted_class}")
                    st.write(f"**Suggested Remedy:** {remedy}")
                elif remedy=='healthy':
                    st.write("The plant appears to be healthy with no visible disease symptoms.")
                else:
                    st.write("No remedy information available for this disease. Please consult an expert for further action.")
        else:
            st.warning("Please upload an image first.")
