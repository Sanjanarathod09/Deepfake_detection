import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Custom CSS for background image
background_image_url = 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.FRFSaw6KRROkFMSy-4h-YgHaEK%26pid%3DApi&f=1&ipt=2b144e2e933df293f634734ae76456d3723d370f156c68c43c7f80b96df09ae9&ipo=images'

page_bg_img = f"""
<style>
.stApp {{
    background-image: url("{background_image_url}");
    background-size: cover;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

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
