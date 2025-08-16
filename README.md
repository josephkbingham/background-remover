# 🪄 Background Remover (rembg + U²-Net)

A simple Streamlit app to remove image backgrounds using the [rembg](https://github.com/danielgatis/rembg) library and U²-Net model.

## Features
- Upload images (PNG, JPG, JPEG, WEBP)
- Remove background with U²-Net
- Optional alpha matting for better edges
- Choose output background: Transparent, White, Black, or Alpha only
- Download the processed image

## Usage
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided local URL in your browser.

## Screenshot
![screenshot](screenshot.png)

---

Made with ❤️ using [Streamlit](https://streamlit.io/) and [rembg](https://github.com/danielgatis/rembg).
