import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Dog & Cat Detector",
    layout="centered"
)

st.title("🐶🐱 Dog & Cat Detection")

@st.cache_resource
def load_model():
    return YOLO("my_model.pt")  # make sure this file exists in repo

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect"):
        with st.spinner("Running detection..."):
            results = model.predict(
                source=img_array,
                conf=0.25,
                device="cpu"   # VERY IMPORTANT for Streamlit Cloud
            )

            annotated = results[0].plot()

        st.success("Detection complete!")
        st.image(annotated, caption="Detection Result", use_container_width=True)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            st.subheader("📦 Detected Objects")
            for cls in boxes.cls.tolist():
                st.write(f"- {model.names[int(cls)]}")
        else:
            st.warning("No dogs or cats detected.")
