
# � Background Remover (BiRefNet, Streamlit)

An advanced Streamlit app for removing image backgrounds using [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) models. Supports NVIDIA, AMD ROCm (Linux), and AMD DirectML (Windows) GPUs.

## Features
- Upload or URL input for images (PNG, JPG, JPEG, WEBP)
- BiRefNet model picker (general, HR, matting, dynamic)
- Aspect-ratio–preserving letterboxing (no distortion)
- Soft alpha by default, optional threshold + feathering
- GPU acceleration: CUDA (NVIDIA), ROCm (AMD/Linux), DirectML (AMD/Windows)
- Download transparent PNG result

## Installation

### A) Linux + AMD ROCm (recommended for AMD)
```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision torchaudio
pip install streamlit pillow transformers timm requests einops kornia
```
Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"` → should print `True` and a HIP version.

### B) Windows + AMD via DirectML (experimental)
```bash
pip install torch-directml
pip install streamlit pillow torchvision transformers timm requests einops kornia
```
Note: Autocast/mixed precision is disabled on DirectML; performance varies by GPU and driver.

### C) NVIDIA (Linux/Windows)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install streamlit pillow transformers timm requests einops kornia
```

## Usage
```bash
streamlit run app.py
```
Then open the provided local URL in your browser.

## Screenshot
![screenshot](screenshot.png)

---

Made with ❤️ using [Streamlit](https://streamlit.io/) and [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet).
