import os
import glob
import json
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Fruit Classifier (12433773)", layout="centered")


def find_model_and_indices():
    folder = "outputs_efficientnet"
    model = glob.glob(os.path.join(folder, "best_model*.h5"))
    if not model:
        return None, None
    return model[0], os.path.join(folder, "class_indices.json")


@st.cache_resource
def load_everything():
    model_path, class_path = find_model_and_indices()
    if model_path is None:
        return None, None, None

    model = load_model(model_path)

    with open(class_path, "r") as f:
        class_map = json.load(f)

    idx2class = {int(v): k for k, v in class_map.items()}
    return model, idx2class, model_path


model, idx2class, model_path = load_everything()

st.title("🍎 Fruit Classification Demo — 12433773")

if model is None:
    st.error("Model not loaded")
    st.stop()

st.write(f"**Loaded model:** `{model_path}`")
st.write(f"**Number of classes:** {len(idx2class)}")

uploaded = st.file_uploader("Upload a fruit image", type=["png", "jpg", "jpeg"])

# ✅ RGB PIPELINE
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", width=300)

    img = img.resize((224, 224))
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    label = idx2class[pred_idx]
    st.success(f"✅ Prediction: **{label}**")
    st.write(f"**Confidence:** {confidence:.2f}%")
