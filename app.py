import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# ---------------------------------------------------------------------
# Load model and metadata
# Cached so the ~125MB model file is loaded once per session rather
# than on every user interaction / rerun.
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_and_metadata():
    model = tf.keras.models.load_model("malaria_cnn_karl.h5")
    with open("malaria_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, metadata

model, metadata = load_model_and_metadata()

CLASS_NAMES = metadata["class_names"]
IMG_SIZE = tuple(metadata["img_size"])   # (height, width)
THRESHOLD = metadata["threshold"]

st.set_page_config(page_title="Malaria Detection", page_icon="🦟")

st.title("🦟 Malaria Detection System")
st.write("Upload a microscopic blood smear image.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # PIL's resize() takes (width, height); IMG_SIZE is stored as
        # (height, width), so the order is flipped here to match.
        img = image.resize((IMG_SIZE[1], IMG_SIZE[0]))
        img = np.array(img).astype(np.float32)

        # No manual rescaling here: the model itself includes a
        # Rescaling(1./255) layer (see resize_rescale in the training
        # notebook), so it expects raw 0-255 pixel values as input.
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)[0][0]

        if prediction >= THRESHOLD:
            predicted_class = CLASS_NAMES[1]
        else:
            predicted_class = CLASS_NAMES[0]

        confidence = prediction if prediction >= THRESHOLD else 1 - prediction

        st.subheader("Prediction")
        st.success(predicted_class)

        st.subheader("Confidence")
        st.write(f"{confidence*100:.2f}%")

    except Exception as e:
        st.error(f"Could not process this image: {e}")
