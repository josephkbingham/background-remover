"""
Streamlit Background Removal App (BiRefNet) — CPU/Streamlit Cloud, NVIDIA, AMD ROCm, AMD (Windows) DirectML, or Remote HF Endpoint

Key upgrades for AMD users:
- **ROCm (Linux):** Works out of the box with this script if you install the ROCm build of PyTorch. `torch.cuda.is_available()` will return True and the device string is still "cuda" (even on AMD under ROCm).
- **DirectML (Windows):** Optional fallback using `torch-directml`. If ROCm/"cuda" is not available but DirectML is, the app runs on `DirectML` with minimal changes. (Autocast is disabled on DirectML.)

Other improvements retained / added:
- Aspect‑ratio–preserving letterboxing (unpadding to original size)
- Model picker (general, HR, matting, dynamic variants)
- Soft alpha by default, optional threshold + feathering
- URL or file input, previews, and a Download Transparent PNG button
- **NEW:** Optional **Hugging Face Inference Endpoint** client (no local model). Provide your endpoint URL + token and the app will call it and composite the returned mask/result.
- **NEW:** Endpoint return-type toggle (mask vs final RGBA) so binary responses are interpreted correctly.
- **NEW:** Built‑in diagnostics & unit tests you can run from the sidebar.

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
"""

from __future__ import annotations
import io
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import requests
import base64
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

# Default: read HF endpoint config from env/secrets if present
DEFAULT_ENDPOINT_URL = os.environ.get("HF_ENDPOINT_URL", "")
DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", "")

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

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"  # 8-byte PNG signature

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
# Inference Pipeline (Local model path)
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
# Remote HF Inference Endpoint client
# -------------------------

def numeric_mask_to_png_bytes(mask_list) -> bytes:
    """Convert a nested list [[floats]] in [0,1] to a PNG (L) byte stream."""
    import numpy as np
    arr = np.array(mask_list, dtype=np.float32)
    arr = (np.clip(arr, 0, 1) * 255).astype("uint8")
    im = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def call_hf_endpoint(
    image: Image.Image,
    endpoint_url: str,
    hf_token: str,
    timeout: int = 120,
    debug: bool = False,
    source_url: Optional[str] = None,
) -> Tuple[Optional[bytes], Optional[Image.Image]]:
    """Call a HF Inference Endpoint trying several payload schemas; return mask bytes or RGBA image.
    Fallback strategies help when the server expects a different JSON shape.
    """
    # Prepare base64 image
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    b64_string = base64.b64encode(img_bytes).decode("utf-8")

    # Candidate payload builders (label, payload)
    candidates = []
    # If URL provided, try URL-first variants (some handlers only accept a string URL)
    if source_url:
        candidates.append(("inputs.url", {"inputs": source_url}))                # simple string URL
        candidates.append(("inputs.image_url", {"inputs": {"image_url": source_url}}))
        candidates.append(("url", {"url": source_url}))
        candidates.append(("inputs.image.url", {"inputs": {"image": {"url": source_url}}}))
    # Base64 variants
    candidates.append(("inputs.image", {"inputs": {"image": b64_string}}))      # nested image base64
    candidates.append(("inputs.image.bytes", {"inputs": {"image": {"bytes": b64_string}}}))
    candidates.append(("inputs-base64", {"inputs": b64_string}))                 # raw base64 string
    candidates.append(("image", {"image": b64_string}))                          # flat image key
    candidates.append(("inputs.bytes", {"inputs": {"bytes": b64_string}}))      # bytes key
    candidates.append(("inputs.data", {"inputs": {"data": b64_string}}))        # data key

    headers = {
        "Authorization": f"Bearer {hf_token}" if hf_token else "",
        "Accept": "image/png,application/json",
        "Content-Type": "application/json",
    }

    errors = []  # (label, status, snippet)
    last_resp = None
    for label, payload in candidates:
        try:
            resp = requests.post(
                endpoint_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except Exception as ex:
            errors.append((label, "EXC", str(ex)[:160]))
            continue
        last_resp = resp
        if resp.ok:
            ctype = resp.headers.get("Content-Type", "")
            content = resp.content
            if ctype.startswith("image/"):
                return content, None
            # JSON path
            if ctype.startswith("application/json"):
                try:
                    js = resp.json()
                except Exception as ex:
                    errors.append((label, resp.status_code, f"JSON parse error: {ex}"))
                    continue
                if debug:
                    st.code({"attempt": label, **{k: (str(v)[:80] + '…' if isinstance(v, str) and len(v) > 80 else v) for k, v in list(js.items())[:6]}})
                # base64 keys
                for key in ("mask_png_b64", "image_png_b64", "mask", "image"):
                    val = js.get(key)
                    if isinstance(val, str):
                        try:
                            raw = base64.b64decode(val)
                        except Exception:
                            continue
                        if raw[:8] == PNG_MAGIC:
                            return raw, None
                        try:
                            im = Image.open(io.BytesIO(raw))
                            return None, im.convert("RGBA")
                        except Exception:
                            pass
                if isinstance(js.get("mask"), list):
                    try:
                        return numeric_mask_to_png_bytes(js["mask"]), None
                    except Exception as ex:
                        errors.append((label, resp.status_code, f"mask convert fail: {ex}"))
                        continue
                # If success but unrecognized JSON, record snippet and continue
                errors.append((label, resp.status_code, f"unhandled JSON keys: {list(js.keys())[:5]}"))
                continue
            else:
                errors.append((label, resp.status_code, f"Unexpected Content-Type {ctype}"))
                continue
        else:
            # store error snippet
            snippet = resp.text[:160].replace('\n', ' ')
            errors.append((label, resp.status_code, snippet))
            continue

    # All attempts failed
    detail = " | ".join(f"{lab}:{st}:{snip}" for lab, st, snip in errors)
    raise RuntimeError(
        "Endpoint attempts failed. Details: " + detail + "\n" +
        "Adjust server to accept one of: {'inputs': {'image': b64}}, {'inputs': b64}, {'image': b64}, or provide a public URL."
    )
# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="BiRefNet Background Removal", layout="wide")
st.title("🧼 Background Removal (BiRefNet)")

with st.sidebar:
    st.header("Remote Inference (HF Endpoint)")
    use_endpoint = st.checkbox("Use HF Inference Endpoint (no local model)", value=True)
    endpoint_url = st.text_input("Endpoint URL", value=DEFAULT_ENDPOINT_URL or "https://e76b464otz1tnrxm.us-east-1.aws.endpoints.huggingface.cloud")
    hf_token = st.text_input("HF Token", value=st.secrets.get("HF_TOKEN", DEFAULT_HF_TOKEN), type="password")
    endpoint_return = st.selectbox("Endpoint returns…", ["mask_png", "rgba_png"], index=0,
                                   help="If your endpoint returns the final composited cutout, choose RGBA.")
    endpoint_debug = st.checkbox("Endpoint debug", value=False, help="Show parsed JSON key snippet when calling endpoint.")
    st.caption("Tip: put HF_TOKEN in Streamlit Secrets for security.")

    st.header("Model & Inference Settings (Local model mode)")
    model_label = st.selectbox("Model", list(MODEL_CHOICES.keys()), index=2)
    repo_id = MODEL_CHOICES[model_label]

    force_cpu = st.checkbox("Force CPU (Streamlit Cloud)", value=DEFAULT_FORCE_CPU,
                            help="Cloud has no GPU. Disable locally if you have CUDA/ROCm/DirectML.")
    if (force_cpu and DEVICE_LABEL != "cpu") or ((not force_cpu) and DEVICE_LABEL == "cpu" and torch.cuda.is_available()):
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

    st.header("Mask Post‑processing (both modes)")
    soft_mask = st.checkbox("Soft alpha (no threshold)", value=True)
    threshold = None if soft_mask else st.slider("Threshold", 0, 255, 128)
    feather = st.slider("Feather radius (px)", 0, 20, 2)
    invert = st.checkbox("Invert mask (keep background)", value=False)

    st.header("Diagnostics & Unit Tests")
    run_tests = st.checkbox("Run internal tests", value=False)

# Load model (cached) — only when not using endpoint
PINNED_REV = "main"  # replace with a commit hash from the HF model card for reproducibility
model = None
if not use_endpoint:
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

# Optional: run internal tests
if run_tests:
    try:
        # Test 1: PNG signature check
        assert PNG_MAGIC == b"\x89PNG\r\n\x1a\n"
        malformed = b"\x89PNG\n\x1a\n"  # wrong (CR missing)
        st.write("Test 1a (correct signature)", PNG_MAGIC[:4])
        st.write("Test 1b (malformed signature) ->", malformed[:8] == PNG_MAGIC)

        # Test 2: numeric mask to PNG bytes round-trip
        mask_list = [
            [0.0, 0.5],
            [1.0, 0.25],
        ]
        png_bytes = numeric_mask_to_png_bytes(mask_list)
        assert png_bytes[:8] == PNG_MAGIC
        img_mask = Image.open(io.BytesIO(png_bytes)).convert("L")
        st.write("Test 2 (numeric mask PNG size)", img_mask.size)
        st.success("All tests passed.")
    except Exception as e:
        st.exception(e)

# Run
if img is not None:
    c1, c2 = st.columns(2)
    with st.spinner("Running…"):
        try:
            t0 = time.perf_counter()
            if use_endpoint:
                # Call remote endpoint
                payload_bytes, _ = call_hf_endpoint(img, endpoint_url, hf_token, debug=endpoint_debug)
                if payload_bytes is None:
                    raise RuntimeError("Endpoint returned no bytes.")
                if endpoint_return == "rgba_png":
                    # Treat bytes as final RGBA image
                    rgba = Image.open(io.BytesIO(payload_bytes)).convert("RGBA")
                    result = rgba
                    mask = None
                else:
                    # Treat bytes as a PNG mask
                    mask = Image.open(io.BytesIO(payload_bytes)).convert("L")
                    result = compose_alpha(
                        img_rgb=img,
                        mask=mask,
                        threshold=threshold,
                        feather_radius=feather,
                        invert=invert,
                    )
            else:
                # Local model path
                mask = run_birefnet(
                    img_rgb=img,
                    model=model,
                    device_label=DEVICE_LABEL,
                    device_obj=DEVICE_OBJ,
                    input_side=input_side,
                    use_letterbox=use_letterbox,
                    use_autocast=use_autocast,
                )
                result = compose_alpha(
                    img_rgb=img,
                    mask=mask,
                    threshold=threshold,
                    feather_radius=feather,
                    invert=invert,
                )
            t1 = time.perf_counter()
        except Exception as e:
            st.exception(e)
            st.stop()

    with c1:
        st.markdown("**Original**")
        st.image(img, use_column_width=True)
        if (not use_endpoint) or (use_endpoint and 'mask' in locals() and mask is not None):
            st.markdown("**Mask (preview)**")
            st.image(mask, use_column_width=True)

    with c2:
        st.markdown("**Result (RGBA)**")
        st.image(result, use_column_width=True)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Transparent PNG",
            data=buf.getvalue(),
            file_name="background_removed.png",
            mime="image/png",
        )

    if use_endpoint:
        st.info(f"Endpoint: {endpoint_url} • Total latency: {(t1 - t0):.3f}s")
    else:
        st.info(
            f"Device: {DEVICE_LABEL} • Total: {(t1 - t0):.3f}s"
        )
else:
    st.warning("Upload a file or paste an image URL to begin.")
