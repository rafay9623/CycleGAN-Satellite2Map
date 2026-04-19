import hashlib
import io
import os
import urllib.error
import urllib.request

import numpy as np
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from PIL import Image, ImageEnhance
from torchvision import transforms

from model import Generator

_HF_ACCESS_ERRORS = (RepositoryNotFoundError,)
try:
    from huggingface_hub.errors import EntryNotFoundError, GatedRepoError

    _HF_ACCESS_ERRORS = (RepositoryNotFoundError, GatedRepoError, EntryNotFoundError)
except ImportError:  # pragma: no cover
    pass

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(page_title="CycleGAN: Satellite ↔ Map", layout="wide")
st.title("🛰️ CycleGAN: Satellite ↔ Map Translation")
st.markdown("Upload a satellite image and translate it to a map — powered by CycleGAN.")

# ─────────────────────────────────────────
# Load models from HuggingFace (cached)
# ─────────────────────────────────────────
_DEFAULT_HF_REPO = "adeelumar17/cyclegan"
_APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _secret_or_env(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    if v:
        return v
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return default


def _hf_token() -> str | None:
    """Private or gated HF repos need a read token (Streamlit: App settings → Secrets)."""
    return _secret_or_env("HF_TOKEN")


def _hf_repo_id() -> str:
    """Override default HF repo via env or secrets, e.g. your own public model repo."""
    return _secret_or_env("HF_REPO_ID", _DEFAULT_HF_REPO) or _DEFAULT_HF_REPO


def _ckpt_direct_urls() -> tuple[str | None, str | None]:
    """Direct HTTPS links to the two .pth files (any public host). Bypasses Hugging Face."""
    return _secret_or_env("CKPT_G_AB_URL"), _secret_or_env("CKPT_G_BA_URL")


def _local_checkpoint(relative_path: str) -> str | None:
    path = os.path.join(_APP_DIR, relative_path.replace("/", os.sep))
    return path if os.path.isfile(path) else None


def _download_ckpt_url(url: str, filename_hint: str) -> str:
    """Download a weight file once and reuse from `.ckpt_cache/` under the app directory."""
    cache_dir = os.path.join(_APP_DIR, ".ckpt_cache")
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    dest = os.path.join(cache_dir, f"{digest}_{filename_hint}")
    if os.path.isfile(dest) and os.path.getsize(dest) > 8192:
        return dest
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; CycleGAN-Streamlit/1.0)"},
    )
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        if len(data) < 8192:
            raise ValueError(
                f"File is only {len(data)} bytes — the URL may point to a web page instead "
                "of the raw `.pth` file. Use a **direct download** link."
            )
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return dest


@st.cache_resource
def load_models(
    repo_id: str,
    hf_token: str | None,
    ckpt_g_ab_url: str | None,
    ckpt_g_ba_url: str | None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def resolve_weight(hf_filename: str, direct_url: str | None) -> str:
        local = _local_checkpoint(hf_filename)
        if local:
            return local
        if direct_url:
            return _download_ckpt_url(direct_url, os.path.basename(hf_filename))
        return hf_hub_download(
            repo_id=repo_id,
            filename=hf_filename,
            token=hf_token,
        )

    G_AB_path = resolve_weight("checkpoints/G_AB_epoch35.pth", ckpt_g_ab_url)
    G_BA_path = resolve_weight("checkpoints/G_BA_epoch35.pth", ckpt_g_ba_url)

    G_AB = Generator()
    G_BA = Generator()

    def _torch_load(path: str):
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=device)

    G_AB.load_state_dict(_torch_load(G_AB_path))
    G_BA.load_state_dict(_torch_load(G_BA_path))

    G_AB.eval()
    G_BA.eval()

    return G_AB.to(device), G_BA.to(device), device


repo_id = _hf_repo_id()
hf_token = _hf_token()
url_ab, url_ba = _ckpt_direct_urls()
if (url_ab and not url_ba) or (not url_ab and url_ba):
    st.error(
        "Set **both** `CKPT_G_AB_URL` and `CKPT_G_BA_URL` in secrets (direct HTTPS links to each "
        "`.pth` file), or leave both unset to use Hugging Face instead."
    )
    st.stop()

def _show_weight_setup_help(download_detail: str | None = None) -> None:
    st.error(
        f"Hub repo **`{_DEFAULT_HF_REPO}`** is not readable without credentials (private or restricted). "
        "Add **Secrets** on Streamlit Cloud (*Manage app* → *Secrets*) using one of the options below."
    )
    with st.expander("Step-by-step: fix weight loading", expanded=True):
        st.markdown(
            "This app needs **`checkpoints/G_AB_epoch35.pth`** and **`checkpoints/G_BA_epoch35.pth`** "
            "from training."
        )
        st.markdown("### A. Hugging Face read token (fastest)")
        st.markdown(
            "1. Open [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create "
            "a token with **read** access.\n"
            "2. In Streamlit: **Manage app** (lower right) → **Secrets**.\n"
            "3. Paste exactly (replace with your token):"
        )
        st.code('HF_TOKEN = "hf_paste_your_read_token_here"', language="toml")
        st.caption("Redeploy or reboot the app after saving secrets.")

        st.markdown("### B. Your own public Hub repo")
        st.markdown(
            "Upload both `.pth` files under `checkpoints/` in a **public** model/dataset repo, then set:"
        )
        st.code('HF_REPO_ID = "your_username/your_repo"', language="toml")

        st.markdown("### C. Direct file URLs (no Hub)")
        st.markdown("Host each file anywhere with a **direct** `https://` download link, then set **both**:")
        st.code(
            'CKPT_G_AB_URL = "https://…/G_AB_epoch35.pth"\nCKPT_G_BA_URL = "https://…/G_BA_epoch35.pth"',
            language="toml",
        )

        st.markdown("### D. Ship weights in Git")
        st.markdown(
            "Commit both files under `checkpoints/` in this repository (use **Git LFS** if GitHub rejects "
            "large files). No secrets needed if the files are in the deployed branch."
        )
        if download_detail:
            st.markdown("---")
            st.markdown("**Last download error**")
            st.text(download_detail)


try:
    spin_msg = (
        "Loading models from direct URLs…"
        if (url_ab and url_ba)
        else f"Loading models (Hub: `{repo_id}`)…"
    )
    with st.spinner(spin_msg):
        G_AB, G_BA, device = load_models(repo_id, hf_token, url_ab, url_ba)
except _HF_ACCESS_ERRORS:
    _show_weight_setup_help()
    st.stop()
except (urllib.error.URLError, ValueError, OSError) as e:
    _show_weight_setup_help(str(e))
    st.stop()
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
with st.sidebar.expander("Weights & deployment"):
    st.caption(
        f"Active Hub repo: `{repo_id}`. Default `{_DEFAULT_HF_REPO}` needs **HF_TOKEN** unless you "
        "set **HF_REPO_ID**, **CKPT_*_URL** pairs, or commit `checkpoints/*.pth`. See "
        "`.streamlit/secrets.toml.example`."
    )

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
            st.image(input_image, use_container_width=True)
        with col2:
            st.subheader(output_label)
            st.image(output_image, use_container_width=True)
        with col3:
            st.subheader(rec_label)
            st.image(rec_image, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(input_image, use_container_width=True)
        with col2:
            st.subheader(output_label)
            st.image(output_image, use_container_width=True)

    # ── Download button ──
    st.markdown("---")
    _png_buf = io.BytesIO()
    output_image.save(_png_buf, format="PNG")
    st.download_button(
        label="⬇️ Download Output Image",
        data=_png_buf.getvalue(),
        file_name="translated_output.png",
        mime="image/png",
    )
