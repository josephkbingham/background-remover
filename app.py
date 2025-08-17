"""
Streamlit Background Removal App (BiRefNet) — CPU/Streamlit Cloud, NVIDIA, AMD ROCm, and AMD (Windows) DirectML

Key upgrades for AMD users:
- **ROCm (Linux):** Works out of the box with this script if you install the ROCm build of PyTorch. `torch.cuda.is_available()` will return True and the device string is still "cuda" (even on AMD under ROCm).
- **DirectML (Windows):** Optional fallback using `torch-directml`. If ROCm/"cuda" is not available but DirectML is, the app runs on `DirectML` with minimal changes. (Autocast is disabled on DirectML.)

Other improvements retained:
- Aspect‑ratio–preserving letterboxing (unpadding to original size)
- Model picker (general, HR, matting, dynamic variants)
- Soft alpha by default, optional threshold + feathering
- URL or file input, previews, and a Download Transparent PNG button

Requirements (for Streamlit Cloud & local):

**0) Streamlit Cloud (no GPU)** — recommended pins
Create a `requirements.txt` like this (CPU-only wheels to avoid accidental CUDA packages):
```
streamlit==1.48.1
transformers==4.55.2
# CPU-only Torch + torchvision; pick a pair supported by Streamlit Cloud's Python (3.11 is safest)
torch==2.4.0
torchvision==0.19.0
pillow==11.0.0
requests==2.32.3
timm==0.9.16
```
Add `runtime.txt` with:
```
3.11
```
This prevents Python 3.13 edge-cases and keeps wheels available.

**A) Linux + AMD ROCm (recommended for AMD)**
```
# Install ROCm-enabled PyTorch (adjust rocm version to your setup)
pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision torchaudio
pip install streamlit pillow transformers timm requests
```
Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"` → should print `True` and a HIP version.

**B) Windows + AMD via DirectML (experimental)**
```
pip install torch-directml
pip install streamlit pillow torchvision transformers timm requests
```
Note: Autocast/mixed precision is disabled on DirectML; performance varies by GPU and driver.

Run:
```
streamlit run streamlit_birefnet_bg_removal.py
```
```
streamlit run streamlit_birefnet_bg_removal.py
```
"""

from __future__ import annotations
import io
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import requests
import streamlit as st
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Try to enable DirectML on Windows if CUDA/ROCm is not available
try:
    import torch_directml as tdm  # type: ignore
    HAS_DML = True
except Exception:
    tdm = None
    HAS_DML = False

# -------------------------
# Configuration & Utilities
# -------------------------

torch.set_float32_matmul_precision("high")  # fast matmul where possible

MODEL_CHOICES = {
    "BiRefNet (general)": "ZhengPeng7/BiRefNet",
    "BiRefNet_HR (hi-res general)": "ZhengPeng7/BiRefNet_HR",
    "BiRefNet_HR-matting (best for soft edges)": "ZhengPeng7/BiRefNet_HR-matting",
    "BiRefNet_dynamic (arbitrary size)": "ZhengPeng7/BiRefNet_dynamic",
    "BiRefNet_dynamic-matting": "ZhengPeng7/BiRefNet_dynamic-matting",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@dataclass
class LetterboxInfo:
    new_w: int
    new_h: int
    pad_left: int
    pad_top: int
    padded_w: int
    padded_h: int


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return img


def to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    raise TypeError("Unsupported image type; must be PIL.Image.Image")


def letterbox(img: Image.Image, target_side: int = 1024, pad_color=(0, 0, 0)) -> Tuple[Image.Image, LetterboxInfo]:
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Invalid image dimensions")
    scale = min(target_side / w, target_side / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    resized = img.resize((new_w, new_h), Image.BICUBIC)
    padded = Image.new("RGB", (target_side, target_side), pad_color)
    pad_left = (target_side - new_w) // 2
    pad_top = (target_side - new_h) // 2
    padded.paste(resized, (pad_left, pad_top))

    info = LetterboxInfo(
        new_w=new_w,
        new_h=new_h,
        pad_left=pad_left,
        pad_top=pad_top,
        padded_w=target_side,
        padded_h=target_side,
    )
    return padded, info


def unletterbox(mask: Image.Image, info: LetterboxInfo, out_size: Tuple[int, int]) -> Image.Image:
    x0, y0 = info.pad_left, info.pad_top
    x1, y1 = x0 + info.new_w, y0 + info.new_h
    cropped = mask.crop((x0, y0, x1, y1))
    return cropped.resize(out_size, Image.BICUBIC)


def make_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

# -------------------------
# Device selection (CUDA/ROCm, DirectML, CPU)
# -------------------------

def pick_device(force_cpu: bool = False) -> Tuple[str, object]:
    """Return (device_label, device_obj).
    If force_cpu=True, always return CPU (useful for Streamlit Cloud).
    """
    if force_cpu:
        return "cpu", "cpu"
    if torch.cuda.is_available():
        return "cuda", "cuda"
    if HAS_DML:
        return "directml", tdm.device()
    return "cpu", "cpu"

# Force CPU by default on Streamlit Cloud (set via env or sidebar)
DEFAULT_FORCE_CPU = bool(os.environ.get("STREAMLIT_RUNTIME") or os.environ.get("STREAMLIT_SERVER_ENABLED"))
DEVICE_LABEL, DEVICE_OBJ = pick_device(force_cpu=DEFAULT_FORCE_CPU)

# -------------------------
# Model Loader (cached)
# -------------------------

@st.cache_resource(show_spinner=True)
def load_model(repo_id: str, device_obj, pin_revision: str | None = None):
    # Pin a revision to avoid unexpected upstream code changes on Cloud
    kwargs = {"trust_remote_code": True}
    if pin_revision:
        kwargs["revision"] = pin_revision
    model = AutoModelForImageSegmentation.from_pretrained(
        repo_id, **kwargs
    )
    model.eval()
    model.to(device_obj)
    return model

# -------------------------
# Inference Pipeline
# -------------------------

def run_birefnet(
    img_rgb: Image.Image,
    model,
    device_label: str,
    device_obj,
    input_side: int = 1024,
    use_letterbox: bool = True,
    use_autocast: bool = True,
) -> Image.Image:
    """Return a soft mask (PIL, single-channel L) of foreground probabilities [0..255]."""
    orig_w, orig_h = img_rgb.size

    if use_letterbox:
        proc_img, linfo = letterbox(img_rgb, target_side=input_side)
        tx_size = input_side
    else:
        w, h = img_rgb.size
        scale = input_side / max(w, h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        proc_img = img_rgb.resize((new_w, new_h), Image.BICUBIC)
        tx_size = max(proc_img.size)

    transform = make_transform(tx_size)
    x = transform(proc_img).unsqueeze(0)

    # move to device
    if device_label == "directml":
        x = x.to(device_obj)
    else:
        x = x.to(device_obj)

    # Predict
    with torch.no_grad():
        if device_label == "cuda" and use_autocast:
            amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                preds = model(x)[-1]
        else:
            preds = model(x)[-1]

    prob = preds.sigmoid().float().cpu()[0].squeeze()  # H x W in [0,1]
    mask_pil = transforms.ToPILImage()(prob)

    if use_letterbox:
        mask_pil = mask_pil.resize((input_side, input_side), Image.BICUBIC)
        mask_pil = unletterbox(mask_pil, linfo, (orig_w, orig_h))
    else:
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.BICUBIC)

    return mask_pil


def compose_alpha(
    img_rgb: Image.Image,
    mask: Image.Image,
    threshold: Optional[int] = None,
    feather_radius: int = 0,
    invert: bool = False,
) -> Image.Image:
    m = mask
    if feather_radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    if invert:
        m = Image.eval(m, lambda p: 255 - p)

    if threshold is not None:
        t = max(0, min(255, int(threshold)))
        m = m.point(lambda p: 255 if p >= t else 0)

    out = img_rgb.copy().convert("RGBA")
    out.putalpha(m)
    return out

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="BiRefNet Background Removal", layout="wide")
st.title("🧼 Background Removal (BiRefNet)")

with st.sidebar:
    st.header("Model & Inference Settings")
    model_label = st.selectbox("Model", list(MODEL_CHOICES.keys()), index=2)
    repo_id = MODEL_CHOICES[model_label]

    force_cpu = st.checkbox("Force CPU (Streamlit Cloud)", value=DEFAULT_FORCE_CPU,
                            help="Cloud has no GPU. Disable locally if you have CUDA/ROCm/DirectML.")
    if (force_cpu and DEVICE_LABEL != "cpu") or ((not force_cpu) and DEVICE_LABEL == "cpu" and torch.cuda.is_available()):
        # Re-pick device based on toggle
        DEVICE_LABEL, DEVICE_OBJ = pick_device(force_cpu=force_cpu)

    st.write(f"**Device:** {DEVICE_LABEL}")

    input_side = st.slider("Network input size (square)", 512, 1536, 896, step=64,
                           help="Lower sizes save RAM on Cloud; 768–1024 is a good balance.")
    use_letterbox = st.checkbox(
        "Letterbox to square (preserve aspect)",
        value=True,
        help="Recommended. Avoids distortion; mask is unpadded back to original size.",
    )
    use_autocast = st.checkbox(
        "Mixed precision (CUDA only)",
        value=True,
        help="Enabled only on CUDA/ROCm. Disabled on DirectML/CPU.",
    )

    if DEVICE_LABEL != "cuda":
        use_autocast = False

    st.header("Mask Post‑processing")
    soft_mask = st.checkbox("Soft alpha (no threshold)", value=True)
    threshold = None if soft_mask else st.slider("Threshold", 0, 255, 128)
    feather = st.slider("Feather radius (px)", 0, 20, 2)
    invert = st.checkbox("Invert mask (keep background)", value=False)

# Load model (cached)
# Pin to a known-good revision (update this if you want newer features)
PINNED_REV = "main"  # replace with a commit hash from the HF model card for reproducibility
model = load_model(repo_id, DEVICE_OBJ, pin_revision=PINNED_REV)

# Inputs
left, right = st.columns(2)
with left:
    st.subheader("1) Upload an image")
    file = st.file_uploader("JPEG/PNG/WebP", type=["jpg", "jpeg", "png", "webp"])
with right:
    st.subheader("2) Or paste an image URL")
    url = st.text_input("https://...")

img: Optional[Image.Image] = None
if file is not None:
    try:
        img = Image.open(file).convert("RGB")
    except Exception as e:
        st.error(f"Failed to read uploaded image: {e}")
elif url:
    try:
        img = load_image_from_url(url)
    except Exception as e:
        st.error(f"Failed to fetch URL: {e}")

# Run
if img is not None:
    c1, c2 = st.columns(2)
    with st.spinner("Running BiRefNet…"):
        try:
            t0 = time.perf_counter()
            mask = run_birefnet(
                img_rgb=img,
                model=model,
                device_label=DEVICE_LABEL,
                device_obj=DEVICE_OBJ,
                input_side=input_side,
                use_letterbox=use_letterbox,
                use_autocast=use_autocast,
            )
            t1 = time.perf_counter()

            result = compose_alpha(
                img_rgb=img,
                mask=mask,
                threshold=threshold,
                feather_radius=feather,
                invert=invert,
            )
            t2 = time.perf_counter()
        except Exception as e:
            st.exception(e)
            st.stop()

    with c1:
        st.markdown("**Original**")
        st.image(img, use_container_width=True)
        st.markdown("**Predicted Mask (preview)**")
        st.image(mask, use_container_width=True)

    with c2:
        st.markdown("**Result (RGBA)**")
        st.image(result, use_container_width=True)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Transparent PNG",
            data=buf.getvalue(),
            file_name="background_removed.png",
            mime="image/png",
        )

    st.info(
        f"Device: {DEVICE_LABEL} • Inference: {(t1 - t0):.3f}s • Compositing: {(t2 - t1):.3f}s • Total: {(t2 - t0):.3f}s"
    )
else:
    st.warning("Upload a file or paste a URL to begin.")
