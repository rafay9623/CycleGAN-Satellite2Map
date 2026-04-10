import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from huggingface_hub import hf_hub_download
from model import Generator
from huggingface_hub import hf_hub_download, login

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(page_title="CycleGAN: Satellite ↔ Map", layout="wide")
st.title("🛰️ CycleGAN: Satellite ↔ Map Translation")
st.markdown("Upload a satellite image and translate it to a map — powered by CycleGAN.")

# ─────────────────────────────────────────
# Load models from HuggingFace (cached)
# ─────────────────────────────────────────
REPO_ID = "adeelumar17/cyclegan"  # replace with your HF repo

@st.cache_resource
def load_models():
    hf_token = st.secrets["HF_TOKEN"]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G_AB_path = hf_hub_download(
        repo_id="adeelumar17/cyclegan",
        filename="checkpoints/G_AB_epoch50.pth",
        token=hf_token
    )

    G_BA_path = hf_hub_download(
        repo_id="adeelumar17/cyclegan",
        filename="checkpoints/G_BA_epoch50.pth",
        token=hf_token
    )

    G_AB = Generator()
    G_BA = Generator()

    G_AB.load_state_dict(torch.load(G_AB_path, map_location=device))
    G_BA.load_state_dict(torch.load(G_BA_path, map_location=device))

    G_AB.eval()
    G_BA.eval()

    return G_AB.to(device), G_BA.to(device), device

with st.spinner("Loading models from HuggingFace..."):
    G_AB, G_BA, device = load_models()
st.success("Models loaded ✅")

# ─────────────────────────────────────────
# Sidebar — tunable parameters
# ─────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio(
    "Translation Mode",
    ["Satellite → Map", "Map → Satellite"]
)

show_reconstruction = st.sidebar.checkbox("Show Reconstruction", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Output Adjustments")

brightness = st.sidebar.slider("Brightness",    0.5, 2.0, 1.0, step=0.1)
contrast   = st.sidebar.slider("Contrast",      0.5, 2.0, 1.0, step=0.1)
sharpness  = st.sidebar.slider("Sharpness",     0.5, 2.0, 1.0, step=0.1)
image_size = st.sidebar.selectbox("Input Resize", [128, 256, 512], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Model: CycleGAN | Dataset: Satellite-Map | Epochs: 50")

# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────
def preprocess(image, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)  # add batch dimension

def tensor_to_image(tensor):
    img = tensor.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0).numpy()
    img = (img + 1) / 2
    img = img.clip(0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))

def apply_adjustments(image, brightness, contrast, sharpness):
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image

# ─────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")

    # ── Run inference ──
    input_tensor = preprocess(input_image, image_size).to(device)

    with torch.no_grad():
        if mode == "Satellite → Map":
            output_tensor = G_AB(input_tensor)
            rec_tensor    = G_BA(output_tensor)
            output_label  = "Generated Map"
            rec_label     = "Reconstructed Satellite"
        else:
            output_tensor = G_BA(input_tensor)
            rec_tensor    = G_AB(output_tensor)
            output_label  = "Generated Satellite"
            rec_label     = "Reconstructed Map"

    output_image = tensor_to_image(output_tensor)
    rec_image    = tensor_to_image(rec_tensor)

    # apply sidebar adjustments to output only
    output_image = apply_adjustments(output_image, brightness, contrast, sharpness)

    # ── Display ──
    if show_reconstruction:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Input")
            st.image(input_image, use_column_width=True)
        with col2:
            st.subheader(output_label)
            st.image(output_image, use_column_width=True)
        with col3:
            st.subheader(rec_label)
            st.image(rec_image, use_column_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(input_image, use_column_width=True)
        with col2:
            st.subheader(output_label)
            st.image(output_image, use_column_width=True)

    # ── Download button ──
    st.markdown("---")
    output_array = np.array(output_image)
    st.download_button(
        label="⬇️ Download Output Image",
        data=output_image.tobytes(),
        file_name="translated_output.png",
        mime="image/png"
    )
