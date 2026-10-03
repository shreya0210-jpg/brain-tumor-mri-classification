import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# Model path
# --------------------------------------------------

MODEL_PATH = "brain_tumor_mobilenetv2.keras"

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# --------------------------------------------------
# Class names
# IMPORTANT: Same order as train_generator.class_indices
# --------------------------------------------------

class_names = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🧠 Brain Tumor MRI Image Classification")

st.write(
    "Upload a brain MRI image and the trained MobileNetV2 "
    "model will classify it into one of four categories."
)

st.info(
    "This application is developed for educational and project "
    "demonstration purposes. It is not a clinical diagnostic tool."
)

# --------------------------------------------------
# Check whether model exists
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):

    st.error(
        "Model file not found. Please place "
        "'brain_tumor_mobilenetv2.keras' in the same folder as app.py."
    )

    st.stop()

# Load model
model = load_model()

# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    # Predict button
    if st.button("🔍 Predict", type="primary"):

        # Resize image
        image_resized = image.resize((224, 224))

        # Convert to NumPy array
        image_array = np.array(image_resized)

        # Normalize pixel values
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Generate prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Find predicted class
        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        # Confidence
        confidence = predictions[0][predicted_index] * 100

        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Class: "
            f"{predicted_class.replace('_', ' ').title()}"
        )

        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )

        # --------------------------------------------------
        # Display probability for each class
        # --------------------------------------------------

        st.subheader("Class Probabilities")

        for i, class_name in enumerate(class_names):

            probability = float(predictions[0][i])

            st.write(
                f"{class_name.replace('_', ' ').title()}: "
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)