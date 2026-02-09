import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Dog & Cat Detector",
    layout="centered"
)

st.title("Dog & Cat Detection")

@st.cache_resource
def load_model():
    return YOLO("my_model.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Placeholders to keep images persistent
    orig_placeholder = st.empty()
    result_placeholder = st.empty()


    if st.button("🔍 Detect"):
        with st.spinner("Running detection..."):
            results = model(img_array, conf=0.25)
            annotated_frame = results[0].plot()

        st.success("Detection complete!")

        # Show annotated image in separate placeholder
        result_placeholder.image(
            annotated_frame,
            caption="Detection Result",
            use_container_width=True
        )

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            st.subheader("📦 Detected Objects")
            for cls in boxes.cls:
                class_name = model.names[int(cls)]
                st.write(f"- {class_name}")
        else:
            st.warning("No dogs or cats detected.")
