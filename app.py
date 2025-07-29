import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import io

# Set up the page
st.set_page_config(
    page_title="StyloGAN - AI Art Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling tweaks
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# Main app function
def main():
    st.markdown('<h1 class="main-header">🎨 StyloGAN</h1>', unsafe_allow_html=True)

    st.write(
        "Turn your photos into artworks inspired by famous styles! "
        "Upload an image and apply styles like Van Gogh, Monet, or Pop Art."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Style Controls")

        styles = {
            "Van Gogh - Starry Night": "starry",
            "The Scream": "scream",
            "Picasso": "picasso",
            "Monet": "monet",
            "Kandinsky": "kandinsky",
            "Ukiyo-e": "ukiyo",
            "Pop Art": "pop",
            "Watercolor": "watercolor"
        }

        selected = st.selectbox("Choose Style", list(styles.keys()))
        intensity = st.slider("Intensity", 0.1, 2.0, 1.0, 0.1)

        with st.expander("Advanced Options"):
            preserve_color = st.checkbox("Preserve Original Colors", False)
            resolution = st.select_slider("Output Size", [256, 512, 768, 1024], value=512)

    # Image upload
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Your Image")
        uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        if uploaded:
            try:
                img = Image.open(uploaded)
                img = ImageOps.exif_transpose(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                st.image(img, caption="Original", use_container_width=True)
            except Exception as e:
                st.error(f"Couldn’t open the image: {e}")
                img = None
        else:
            img = None

    with col2:
        st.subheader("Styled Output")
        if img:
            if st.button("Generate Art", use_container_width=True):
                with st.spinner("Styling your image..."):
                    styled = apply_style(img, styles[selected], intensity, preserve_color, resolution)
                    if styled:
                        st.image(styled, caption=f"{selected} Style", use_container_width=True)

                        buffer = io.BytesIO()
                        styled.save(buffer, format="PNG")
                        st.download_button(
                            "Download",
                            buffer.getvalue(),
                            file_name="styled_image.png",
                            mime="image/png"
                        )
                    else:
                        st.error("Something went wrong.")
        else:
            st.info("Upload an image to begin.")

# Style application logic
def apply_style(img, style, intensity, preserve, size):
    try:
        img = resize_image(img, size)

        if style == "starry" or style == "scream":
            result = style_van_gogh(img, intensity)
        elif style == "picasso":
            result = style_picasso(img, intensity)
        elif style == "monet":
            result = style_monet(img, intensity)
        elif style == "kandinsky":
            result = style_kandinsky(img, intensity)
        elif style == "ukiyo":
            result = style_ukiyo(img, intensity)
        elif style == "pop":
            result = style_pop(img, intensity)
        elif style == "watercolor":
            result = style_watercolor(img, intensity)
        else:
            result = img

        if preserve:
            result = Image.blend(result, img, 0.5 * intensity)

        return result
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Resize if too large
def resize_image(img, size):
    if max(img.size) > size:
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
    return img

# Stylization functions
def style_van_gogh(img, intensity):
    img = ImageEnhance.Color(img).enhance(1 + 0.5 * intensity)
    img = img.filter(ImageFilter.GaussianBlur(radius=2 * intensity))
    img = ImageEnhance.Contrast(img).enhance(1 + 0.3 * intensity)
    return img

def style_picasso(img, intensity):
    edges = img.filter(ImageFilter.FIND_EDGES)
    quant = img.quantize(colors=max(4, int(16 - intensity * 8))).convert('RGB')
    return Image.blend(quant, edges, 0.3 * intensity)

def style_monet(img, intensity):
    img = img.filter(ImageFilter.GaussianBlur(radius=3 * intensity))
    img = ImageEnhance.Brightness(img).enhance(1 + 0.2 * intensity)
    img = ImageEnhance.Color(img).enhance(1 + 0.3 * intensity)
    return img

def style_kandinsky(img, intensity):
    img = ImageEnhance.Color(img).enhance(1 + 0.8 * intensity)
    img = ImageEnhance.Contrast(img).enhance(1 + 0.6 * intensity)
    return img.filter(ImageFilter.GaussianBlur(radius=intensity))

def style_ukiyo(img, intensity):
    img = ImageOps.posterize(img, int(8 - 2 * intensity))
    img = ImageEnhance.Contrast(img).enhance(1 + 0.5 * intensity)
    edges = img.filter(ImageFilter.FIND_EDGES)
    return Image.blend(img, edges, 0.2 * intensity)

def style_pop(img, intensity):
    img = img.quantize(colors=max(6, int(12 - intensity * 4))).convert('RGB')
    img = ImageEnhance.Color(img).enhance(1 + 1.2 * intensity)
    img = ImageEnhance.Contrast(img).enhance(1 + 0.8 * intensity)
    return img

def style_watercolor(img, intensity):
    for _ in range(int(intensity * 3)):
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    img = ImageEnhance.Color(img).enhance(1 + 0.4 * intensity)
    img = ImageEnhance.Brightness(img).enhance(1 + 0.1 * intensity)
    return img

# Run app
if __name__ == "__main__":
    main()
