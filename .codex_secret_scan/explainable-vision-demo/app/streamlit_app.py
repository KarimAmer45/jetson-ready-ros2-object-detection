from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xai_vision_demo.data import build_transforms  # noqa: E402
from xai_vision_demo.explain import GradCAM, overlay_heatmap  # noqa: E402
from xai_vision_demo.model import create_model, gradcam_target_layer  # noqa: E402


st.set_page_config(page_title="Explainable Vision Demo", layout="wide")


@st.cache_resource
def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = create_model(
        checkpoint["arch"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


st.title("Explainable Vision Demo")

with st.sidebar:
    checkpoint_path = st.text_input("Checkpoint", "runs/surface_resnet18/best_model.pt")
    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])

if not Path(checkpoint_path).exists():
    st.info("Train a model or enter a valid checkpoint path to start inference.")
    st.stop()

if uploaded is None:
    st.info("Upload an image to classify it and generate GradCAM.")
    st.stop()

model, checkpoint = load_model(checkpoint_path)
class_names = checkpoint["class_names"]
image = Image.open(uploaded).convert("RGB")
transform = build_transforms(checkpoint["image_size"], train=False)
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

target_class = int(probs.argmax())
gradcam = GradCAM(model, gradcam_target_layer(model, checkpoint["arch"]))
try:
    heatmap = gradcam(image_tensor, class_index=target_class)
finally:
    gradcam.close()

overlay = overlay_heatmap(image.resize((checkpoint["image_size"], checkpoint["image_size"])), heatmap)
score_table = (
    pd.DataFrame({"class": class_names, "probability": probs})
    .sort_values("probability", ascending=False)
    .reset_index(drop=True)
)

left, right = st.columns([1, 1])
with left:
    st.subheader("Input")
    st.image(image, use_container_width=True)
with right:
    st.subheader("GradCAM")
    st.image(overlay, use_container_width=True)

st.subheader("Prediction")
st.metric("Top class", score_table.loc[0, "class"], f"{score_table.loc[0, 'probability']:.1%}")
st.bar_chart(score_table.set_index("class"))
