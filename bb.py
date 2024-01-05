import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

# Load the pre-trained model
model = load_model('my_model.h5')

# Define the image size for preprocessing
img_size = (150, 150)

# Streamlit App
st.title("Brain Tumor Detection App")

# File uploader widget
uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for the model
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to between 0 and 1
    img_array = preprocess_input(img_array)  # Preprocess for the specific model

    # Make predictions
    prediction = model.predict(img_array)
    class_names = ['No Tumor', 'Tumor']
    # Print raw output values
    st.subheader("Raw Prediction Scores:")
    st.write(prediction)


    # Display the prediction
    st.subheader("Prediction:")
    st.write(f"This image is classified as {class_names[np.argmax(prediction)]}.")

    # Display the probability scores for each class
    #st.subheader("Probability Scores:")
    #st.write({class_names[i]: round(float(prediction[0][i]), 4) for i in range(len(class_names))})
