import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load your YOLOv8n model (replace 'path/to/your/model.pt' with the actual path to your model)
model = YOLO('yolov8n.pt')

st.title("Deepfake Detection")
st.write("Upload an image to classify between real vs deepfake")

# File uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Results")

    # Perform classification
    results = model.predict(image)

    # Get the predicted class
    probs = results[0].probs
    predicted_class_index = probs.top1
    predicted_class_confidence = probs.top1conf

    # Retrieve class names from the model's attribute
    class_names = model.names

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]

    st.write(f"Predicted class: {predicted_class_name}")
    st.write(f"Confidence: {predicted_class_confidence:.2f}")
