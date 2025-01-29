# classify.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('models/cnn_model.keras')

# Define class labels
class_labels = [f"Class {i+1}" for i in range(40)]  # Replace with actual class names if known

# Function to classify images
def classify_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    predictions = model.predict(img_array)
    return predictions
