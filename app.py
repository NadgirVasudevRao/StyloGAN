import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# Configure page
st.set_page_config(
    page_title="Art Style Transfer Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        background-clip: text;
    }
    
    .style-card {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4ecdc4;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎨 Art Style Transfer Studio</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Transform your photos into masterpieces inspired by famous artists! 
    Upload an image and watch it transform into the style of Van Gogh, Picasso, Monet, and more.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Style Controls")
        
        # Style selection
        art_styles = {
            "Van Gogh - Starry Night": "starry_night",
            "Van Gogh - The Scream": "scream", 
            "Picasso - Cubist": "picasso",
            "Monet - Impressionist": "monet",
            "Kandinsky - Abstract": "kandinsky",
            "Japanese Ukiyo-e": "ukiyo_e",
            "Pop Art": "pop_art",
            "Watercolor": "watercolor"
        }
        
        selected_style = st.selectbox(
            "Choose Art Style",
            list(art_styles.keys()),
            help="Select the artistic style to apply to your image"
        )
        
        # Intensity slider
        style_intensity = st.slider(
            "Style Intensity",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Control how strong the artistic effect should be"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            preserve_colors = st.checkbox(
                "Preserve Original Colors",
                value=False,
                help="Keep the original colors while applying style"
            )
            
            output_size = st.select_slider(
                "Output Resolution",
                options=[256, 512, 768, 1024],
                value=512,
                help="Higher resolution takes longer but gives better quality"
            )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a photo to transform into art"
        )
        
        if uploaded_file is not None:
            # Load and display original image with proper orientation
            original_image = load_image_with_orientation(uploaded_file)
            st.image(original_image, caption="Original Image", use_container_width=True)
            
            # Show image info
            st.info(f"Image size: {original_image.size[0]}x{original_image.size[1]} pixels")
    
    with col2:
        st.subheader("🎨 Styled Result")
        
        if uploaded_file is not None:
            if st.button("🚀 Generate Art", type="primary", use_container_width=True):
                with st.spinner(f"Applying {selected_style} style... This may take a moment."):
                    try:
                        # Process the image
                        styled_image = apply_style_transfer(
                            original_image,
                            art_styles[selected_style],
                            style_intensity,
                            preserve_colors,
                            output_size
                        )
                        
                        if styled_image is not None:
                            st.image(styled_image, caption=f"Styled as {selected_style}", use_container_width=True)
                            
                            # Download button
                            img_buffer = io.BytesIO()
                            styled_image.save(img_buffer, format='PNG')
                            img_data = img_buffer.getvalue()
                            
                            st.download_button(
                                label="💾 Download Styled Image",
                                data=img_data,
                                file_name=f"styled_{selected_style.lower().replace(' ', '_')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to generate styled image. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Error during style transfer: {str(e)}")
        else:
            st.info("👆 Upload an image to see the styled result here")
    
    # Style gallery
    st.markdown("---")
    st.subheader("🖼️ Style Gallery")
    st.markdown("Preview of different artistic styles available:")
    
    # Create style preview grid
    style_cols = st.columns(4)
    style_previews = [
        ("Van Gogh", "🌌", "Swirling brushstrokes and vibrant colors"),
        ("Picasso", "🔷", "Geometric shapes and abstract forms"),
        ("Monet", "🌸", "Soft, impressionistic light and color"),
        ("Kandinsky", "🎭", "Bold abstract compositions")
    ]
    
    for i, (style, icon, desc) in enumerate(style_previews):
        with style_cols[i]:
            st.markdown(f"""
            <div class="style-card">
                <h4>{icon} {style}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def apply_style_transfer(content_image, style_name, intensity, preserve_colors, output_size):
    """Apply neural style transfer to the image"""
    try:
        return apply_traditional_style_transfer(content_image, style_name, intensity, preserve_colors, output_size)
    except Exception as e:
        st.error(f"Style transfer error: {str(e)}")
        return None

def apply_traditional_style_transfer(image, style_name, intensity, preserve_colors, output_size):
    """Apply artistic effects using traditional image processing"""
    # Keep image in RGB format to avoid color channel confusion
    rgb_image = np.array(image)
    
    # Resize image if needed
    height, width = rgb_image.shape[:2]
    if max(height, width) > output_size:
        if height > width:
            new_height = output_size
            new_width = int(width * output_size / height)
        else:
            new_width = output_size
            new_height = int(height * output_size / width)
        rgb_image = cv2.resize(rgb_image, (new_width, new_height))
    
    # Apply style-specific effects (modified to work with RGB)
    if style_name == "starry_night" or style_name == "scream":
        result = apply_van_gogh_style(rgb_image, intensity)
    elif style_name == "picasso":
        result = apply_picasso_style(rgb_image, intensity)
    elif style_name == "monet":
        result = apply_monet_style(rgb_image, intensity)
    elif style_name == "kandinsky":
        result = apply_kandinsky_style(rgb_image, intensity)
    elif style_name == "ukiyo_e":
        result = apply_ukiyo_e_style(rgb_image, intensity)
    elif style_name == "pop_art":
        result = apply_pop_art_style(rgb_image, intensity)
    elif style_name == "watercolor":
        result = apply_watercolor_style(rgb_image, intensity)
    else:
        result = apply_van_gogh_style(rgb_image, intensity)
    
    # Preserve original colors if requested
    if preserve_colors:
        result = preserve_original_colors(rgb_image, result, intensity * 0.5)
    
    # Return PIL Image directly from RGB array
    return Image.fromarray(np.uint8(result))

def apply_van_gogh_style(image, intensity):
    """Apply Van Gogh-inspired swirling brushstroke effects"""
    result = image.copy()
    
    # Apply artistic color enhancement
    result = enhance_colors_artistic(result, intensity)
    
    # Create swirling motion blur effect
    blur_amount = int(5 * intensity)
    if blur_amount % 2 == 0:
        blur_amount += 1
    if blur_amount > 1:
        result = cv2.GaussianBlur(result, (blur_amount, blur_amount), 0)
    
    # Add texture
    result = add_canvas_texture(result, intensity * 0.3)
    
    return result

def apply_picasso_style(image, intensity):
    """Apply Picasso-inspired cubist effects"""
    result = image.copy()
    
    # Edge detection for geometric shapes
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Dilate edges to make them more prominent
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=int(intensity))
    
    # Color quantization for flat color areas
    data = result.reshape((-1, 3))
    data = np.float32(data)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = max(4, int(8 - intensity * 2))  # Fewer colors for more abstract look
    
    # Use the correct parameter types for kmeans
    compactness, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    result = result.reshape(image.shape)
    
    # Overlay edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    result = cv2.addWeighted(result, 0.8, edges_colored, 0.2 * intensity, 0)
    
    return result

def apply_monet_style(image, intensity):
    """Apply Monet-inspired impressionist effects"""
    result = image.copy()
    
    # Soft gaussian blur for impressionistic effect
    blur_amount = int(5 * intensity)
    if blur_amount % 2 == 0:
        blur_amount += 1
    if blur_amount > 1:
        result = cv2.GaussianBlur(result, (blur_amount, blur_amount), 0)
    
    # Enhance brightness and saturation
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + intensity * 0.3), 0, 255)  # Saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + intensity * 0.2), 0, 255)  # Brightness
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Add soft texture
    result = add_soft_texture(result, intensity * 0.4)
    
    return result

def apply_kandinsky_style(image, intensity):
    """Apply Kandinsky-inspired abstract effects"""
    result = image.copy()
    
    # Strong color enhancement
    result = enhance_colors_artistic(result, intensity * 1.5)
    
    # Abstract geometric distortion
    height, width = result.shape[:2]
    distortion_strength = intensity * 20
    
    # Create wave distortion
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x_distorted = x + distortion_strength * np.sin(y / 30.0)
    y_distorted = y + distortion_strength * np.cos(x / 30.0)
    
    # Apply distortion
    x_distorted = np.clip(x_distorted, 0, width - 1).astype(np.float32)
    y_distorted = np.clip(y_distorted, 0, height - 1).astype(np.float32)
    
    result = cv2.remap(result, x_distorted, y_distorted, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return result

def apply_ukiyo_e_style(image, intensity):
    """Apply Japanese Ukiyo-e woodblock print effects"""
    result = image.copy()
    
    # High contrast and color quantization
    data = result.reshape((-1, 3))
    data = np.float32(data)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = max(6, int(12 - intensity * 3))
    
    compactness, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    result = result.reshape(image.shape)
    
    # Strong edge detection
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    
    # Overlay edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    result = cv2.subtract(result, edges_colored)
    
    return result

def apply_pop_art_style(image, intensity):
    """Apply Pop Art effects"""
    result = image.copy()
    
    # High contrast
    result = cv2.convertScaleAbs(result, alpha=1.0 + intensity * 0.5, beta=20)
    
    # Color quantization with bright colors
    data = result.reshape((-1, 3))
    data = np.float32(data)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = max(4, int(8 - intensity))
    
    compactness, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Boost bright colors
    centers = np.uint8(np.clip(centers * 1.2, 0, 255))
    result = centers[labels.flatten()]
    result = result.reshape(image.shape)
    
    return result

def apply_watercolor_style(image, intensity):
    """Apply watercolor painting effects"""
    result = image.copy()
    
    # Multiple blur passes with different kernel sizes
    for i in range(int(intensity * 3)):
        kernel_size = 5 + i * 2
        if kernel_size % 2 == 0:
            kernel_size += 1
        result = cv2.bilateralFilter(result, kernel_size, 80, 80)
    
    # Soft color enhancement
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.8 + intensity * 0.4), 0, 255)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Add paper texture
    result = add_paper_texture(result, intensity * 0.5)
    
    return result

def enhance_colors_artistic(image, intensity):
    """Enhance colors in an artistic way"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + intensity * 0.5), 0, 255)  # Saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + intensity * 0.3), 0, 255)  # Value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def add_canvas_texture(image, intensity):
    """Add canvas-like texture"""
    height, width = image.shape[:2]
    texture = np.random.normal(0, intensity * 10, (height, width)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        channel = np.clip(channel + texture, 0, 255)
        image[:, :, c] = channel.astype(np.uint8)
    
    return image

def add_soft_texture(image, intensity):
    """Add soft, organic texture"""
    height, width = image.shape[:2]
    texture = np.random.normal(0, intensity * 5, (height, width)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        channel = np.clip(channel + texture, 0, 255)
        image[:, :, c] = channel.astype(np.uint8)
    
    return image

def add_paper_texture(image, intensity):
    """Add paper-like texture"""
    height, width = image.shape[:2]
    texture = np.random.uniform(-intensity * 15, intensity * 15, (height, width)).astype(np.float32)
    
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        channel = np.clip(channel + texture, 0, 255)
        image[:, :, c] = channel.astype(np.uint8)
    
    return image

def preserve_original_colors(original, styled, blend_factor):
    """Preserve original colors by blending with styled image"""
    # Convert to LAB color space
    original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    styled_lab = cv2.cvtColor(styled, cv2.COLOR_RGB2LAB)
    
    # Keep original A and B channels (color), use styled L channel (lightness)
    result_lab = styled_lab.copy()
    result_lab[:, :, 1] = cv2.addWeighted(original_lab[:, :, 1], blend_factor, styled_lab[:, :, 1], 1 - blend_factor, 0)
    result_lab[:, :, 2] = cv2.addWeighted(original_lab[:, :, 2], blend_factor, styled_lab[:, :, 2], 1 - blend_factor, 0)
    
    # Convert back to RGB
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

def load_image_with_orientation(uploaded_file):
    """Load image and correct EXIF orientation to prevent rotation issues"""
    try:
        image = Image.open(uploaded_file)
        
        # Handle EXIF orientation automatically using ImageOps
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
        except Exception:
            # Fallback: manual EXIF handling if ImageOps fails
            try:
                if hasattr(image, '_getexif'):
                    exif = image._getexif()
                    if exif is not None:
                        orientation = exif.get(0x0112)  # Orientation tag
                        if orientation == 2:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 4:
                            image = image.transpose(Image.FLIP_TOP_BOTTOM)
                        elif orientation == 5:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
                        elif orientation == 6:
                            image = image.rotate(270, expand=True)
                        elif orientation == 7:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)
            except Exception:
                pass  # Use original image if all EXIF handling fails
        
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

if __name__ == "__main__":
    main()