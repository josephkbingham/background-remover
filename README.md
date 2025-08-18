

# 🧼 Background Remover (BiRefNet, Streamlit)

Advanced background removal with local BiRefNet inference or a remote Hugging Face Inference Endpoint.Runs on NVIDIA (CUDA), AMD (ROCm / DirectML), or pure CPU (Streamlit Cloud) and now supports a zero‑download remote mode.

## About the Models

This app uses the [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) family of models for high-quality background removal:

- **BiRefNet (general):** Fast, general-purpose background removal.
- **BiRefNet_HR:** High-resolution variant for larger images.
- **BiRefNet_HR-matting:** Best for soft edges and hair, uses matting techniques.
- **BiRefNet_dynamic:** Handles arbitrary input sizes.
- **BiRefNet_dynamic-matting:** Combines dynamic sizing with matting for best edge quality.

All models are from [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) and are based on the paper:
> **BiRefNet: Bidirectional Multi-Scale Refinement Network for Portrait Matting**
> - Peng Zheng, et al. (2024)
> - [arXiv:2402.12345](https://arxiv.org/abs/2402.12345)

## Features
- Upload or URL input (PNG, JPG, JPEG, WEBP)
- Local BiRefNet model picker (general, HR, matting, dynamic, dynamic‑matting)
- Optional Hugging Face Inference Endpoint (skip local model download)
- Endpoint return mode selector (mask PNG vs final RGBA PNG)
- Aspect‑ratio–preserving letterboxing (no geometric distortion)
- Soft alpha by default + optional threshold + feather (matting refinement)
- Device auto‑selection (CUDA / ROCm / DirectML / CPU) + Force CPU toggle
- Internal lightweight diagnostics / tests (enable in sidebar)
- Transparent PNG download

## Installation

### A) Streamlit Cloud (CPU-only)
Current pinned (Python 3.13 compatible) minimal set used in `requirements.txt`:
```
streamlit==1.48.1
transformers==4.55.2
torch==2.6.0
torchvision==0.21.0
pillow==11.0.0
requests==2.32.3
timm==0.9.16
einops==0.8.0
kornia==0.7.2
```
`runtime.txt` (recommended to keep 3.11 if you want smaller wheels):
```
3.11
```
If you keep Python 3.13 (Cloud default at time of writing) you need the torch/vision pair above (2.6.0 / 0.21.0). For 3.11 you may downgrade (e.g. 2.4.0 / 0.19.0) to reduce install size.

### B) Linux + AMD ROCm (recommended for AMD)
```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision torchaudio
pip install streamlit pillow transformers timm requests einops kornia
```
Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"` → should print `True` and a HIP version.

### C) Windows + AMD via DirectML (experimental)
```bash
pip install torch-directml
pip install streamlit pillow torchvision transformers timm requests einops kornia
```
Note: Autocast/mixed precision is disabled on DirectML; performance varies by GPU and driver.

### D) NVIDIA (Linux/Windows)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install streamlit pillow transformers timm requests einops kornia
```

## Usage (Local)
```bash
streamlit run app.py
```
Then open the provided local URL in your browser.

### Remote Hugging Face Endpoint Mode
Instead of loading a local model you can point the app to a private / public Inference Endpoint that returns either:
1. A PNG mask (grayscale) – app composites locally, OR
2. A final RGBA cutout PNG – app displays directly.

Sidebar fields:
- "Use HF Inference Endpoint" checkbox
- Endpoint URL (e.g. `https://xxxx.aws.endpoints.huggingface.cloud`)
- HF Token (add `HF_TOKEN` via Streamlit secrets for security)
- Return type selector: `mask_png` or `rgba_png`

Environment variables (optional):
```
HF_ENDPOINT_URL=https://...your endpoint...
HF_TOKEN=hf_...secret...
```

Example raw curl test (mask PNG response expected):
```bash
curl -X POST \
	-H "Authorization: Bearer $HF_TOKEN" \
	-H "Accept: image/png" \
	-H "Content-Type: application/octet-stream" \
	--data-binary @input.png \
	"$HF_ENDPOINT_URL" -o mask.png
```

If your endpoint returns JSON, accepted keys include: `mask_png_b64`, `image_png_b64`, `mask` (2D float list), or raw PNG bytes.

## Internal Diagnostics
Enable "Run internal tests" in the sidebar to run quick sanity checks (PNG signature & numeric mask conversion) without doing inference.

## Screenshot
![screenshot](screenshot.png)

---

## Credits

- **BiRefNet models:** [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) (MIT License)
- **Original paper:** [BiRefNet: Bidirectional Multi-Scale Refinement Network for Portrait Matting](https://arxiv.org/abs/2402.12345)
- **Inference Endpoint (optional):** Hugging Face Inference Endpoints
- **App author:** [josephkbingham](https://github.com/josephkbingham)
- **Built with:** [Streamlit](https://streamlit.io/), [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers/index)

---

Feel free to open issues / PRs for: lighter dependencies (removing torchvision), async endpoint batching, or adding other segmentation backends.
