import io
import streamlit as st
from rembg import remove
from PIL import Image

st.set_page_config(page_title="Background Remover", page_icon="🪄")

@st.cache_resource
def get_remover():
    # First call will download model if not present
    return remove

st.title("🪄 Background Remover (rembg + U²-Net)")
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

col1, col2 = st.columns(2)
if uploaded:
    input_bytes = uploaded.read()
    input_img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")

    # Optional knob: alpha matting (good for hair/edges)
    use_alpha_matting = st.checkbox("Use alpha matting (slower, better edges)", value=True)
    bg_choice = st.selectbox("Output background", ["Transparent (PNG)", "White", "Black", "Keep alpha only"])

    remover = get_remover()
    output_bytes = remover(
        input_bytes,
        alpha_matting=use_alpha_matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )

    out = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    if bg_choice != "Transparent (PNG)":
        bg = Image.new("RGBA", out.size, (255, 255, 255, 255) if bg_choice == "White" else (0, 0, 0, 255))
        out = Image.alpha_composite(bg, out)

    with col1:
        st.subheader("Original")
        st.image(input_img, use_container_width=True)
    with col2:
        st.subheader("Result")
        st.image(out, use_container_width=True)

    buf = io.BytesIO()
    fmt = "PNG" if bg_choice == "Transparent (PNG)" else "JPEG"
    out.save(buf, format=fmt)
    st.download_button("Download result", buf.getvalue(), file_name=f"output.{fmt.lower()}", mime=f"image/{fmt.lower()}")
else:
    st.info("Upload an image to begin.")
