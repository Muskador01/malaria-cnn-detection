import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# ---------------------------------------------------------------------
# Load model and metadata
# ---------------------------------------------------------------------
model = tf.keras.models.load_model("malaria_cnn_karl.h5")

with open("malaria_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

CLASS_NAMES = metadata["class_names"]
IMG_SIZE = tuple(metadata["img_size"])   # (151, 136) -> (height, width)
THRESHOLD = metadata["threshold"]

st.set_page_config(page_title="Malaria Detection", page_icon="🦟")

st.title("🦟 Malaria Detection System")
st.write("Upload a microscopic blood smear image.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # PIL's resize takes (width, height), and IMG_SIZE is (height, width),
    # so we flip the order here -- this part was already correct.
    img = image.resize((IMG_SIZE[1], IMG_SIZE[0]))
    img = np.array(img).astype(np.float32)
    # NOTE: no /255.0 here! The model itself has a Rescaling(1./255) layer
    # baked in (see resize_rescale in the notebook), so it expects raw
    # 0-255 pixel values as input. Dividing here as well was double-
    # rescaling every image down to near-zero, which is why every
    # prediction came out the same regardless of the input image.
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]
    st.write("Raw prediction:", prediction)

    if prediction >= THRESHOLD:
        predicted_class = CLASS_NAMES[1]
    else:
        predicted_class = CLASS_NAMES[0]

    confidence = prediction if prediction >= THRESHOLD else 1 - prediction

    st.subheader("Prediction")
    st.success(predicted_class)

    st.subheader("Confidence")
    st.write(f"{confidence*100:.2f}%")
