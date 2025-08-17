

# 🧼 Background Remover (BiRefNet, Streamlit)

An advanced Streamlit app for removing image backgrounds using [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) models. Supports NVIDIA, AMD ROCm (Linux), AMD DirectML (Windows), and CPU (Streamlit Cloud) deployments.

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
- Upload or URL input for images (PNG, JPG, JPEG, WEBP)
- BiRefNet model picker (general, HR, matting, dynamic)
- Aspect-ratio–preserving letterboxing (no distortion)
- Soft alpha by default, optional threshold + feathering
- GPU acceleration: CUDA (NVIDIA), ROCm (AMD/Linux), DirectML (AMD/Windows), or CPU fallback
- Download transparent PNG result

## Installation

### A) Streamlit Cloud (CPU-only, recommended for cloud deploy)
Create a `requirements.txt`:
```
streamlit==1.48.1
transformers==4.55.2
torch==2.4.0
torchvision==0.19.0
pillow==11.0.0
requests==2.32.3
timm==0.9.16
```
And a `runtime.txt`:
```
3.11
```

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

## Usage
```bash
streamlit run app.py
```
Then open the provided local URL in your browser.

## Screenshot
![screenshot](screenshot.png)

---

## Credits

- **BiRefNet models:** [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) (MIT License)
- **Original paper:** [BiRefNet: Bidirectional Multi-Scale Refinement Network for Portrait Matting](https://arxiv.org/abs/2402.12345)
- **App author:** [Your Name or GitHub](https://github.com/josephkbingham)
- **Built with:** [Streamlit](https://streamlit.io/), [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers/index)
