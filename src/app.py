import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/yolov8_aquarium11/weights/best.pt')

# App title
st.title("YOLOv8 Object Detection Demo")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Run inference
    st.write("Detecting...")
    results = model(image)
    # Updated code to handle if `results` is a list
    results_image = results[0].show() if isinstance(results, list) else results.show()
 # Get the image with results

    # Display results
    st.image(results_image, caption='Detected Objects', use_column_width=True)
    
    # Show Metrics (Optional: use static values or calculate live)
    st.write("Metrics:")
    st.write("Precision:", results.metrics.get('precision', 'N/A'))
    st.write("Recall:", results.metrics.get('recall', 'N/A'))
    st.write("mAP50:", results.metrics.get('mAP50', 'N/A'))
    st.write("mAP50-95:", results.metrics.get('mAP50-95', 'N/A'))

