import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import io

# Configure page
st.set_page_config(
    page_title="StyloGAN - AI Art Style Transfer",
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
    st.markdown('<h1 class="main-header">🎨 StyloGAN</h1>', unsafe_allow_html=True)
    
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
                help="Maintain the original color palette while applying the style"
            )
            
            output_size = st.select_slider(
                "Output Resolution",
                options=[256, 512, 768, 1024],
                value=512,
                help="Higher resolution = better quality but slower processing"
            )
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a photo to transform into art"
        )
        
        if uploaded_file is not None:
            try:
                # Load and process the image
                original_image = Image.open(uploaded_file)
                
                # Handle EXIF orientation
                try:
                    original_image = ImageOps.exif_transpose(original_image)
                except:
                    pass
                
                # Convert to RGB if necessary
                if original_image.mode != 'RGB':
                    original_image = original_image.convert('RGB')
                
                st.image(original_image, caption="Original Image", use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                original_image = None
        else:
            original_image = None
    
    with col2:
        st.subheader("🎨 Styled Result")
        
        if original_image is not None:
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

def apply_style_transfer(content_image, style_name, intensity, preserve_colors, output_size):
    """Apply PIL-based style transfer to the image"""
    try:
        # Resize image if needed
        width, height = content_image.size
        if max(width, height) > output_size:
            content_image.thumbnail((output_size, output_size), Image.Resampling.LANCZOS)
        
        # Apply style-specific effects
        if style_name == "starry_night" or style_name == "scream":
            result = apply_van_gogh_style_pil(content_image, intensity)
        elif style_name == "picasso":
            result = apply_picasso_style_pil(content_image, intensity)
        elif style_name == "monet":
            result = apply_monet_style_pil(content_image, intensity)
        elif style_name == "kandinsky":
            result = apply_kandinsky_style_pil(content_image, intensity)
        elif style_name == "ukiyo_e":
            result = apply_ukiyo_e_style_pil(content_image, intensity)
        elif style_name == "pop_art":
            result = apply_pop_art_style_pil(content_image, intensity)
        elif style_name == "watercolor":
            result = apply_watercolor_style_pil(content_image, intensity)
        else:
            result = apply_van_gogh_style_pil(content_image, intensity)
        
        # Preserve original colors if requested
        if preserve_colors:
            result = preserve_original_colors_pil(content_image, result, intensity * 0.5)
        
        return result
        
    except Exception as e:
        st.error(f"Style transfer error: {str(e)}")
        return None

def apply_van_gogh_style_pil(image, intensity):
    """Apply Van Gogh-inspired effects using PIL"""
    # Enhance colors
    enhancer = ImageEnhance.Color(image)
    result = enhancer.enhance(1.0 + intensity * 0.5)
    
    # Add artistic blur
    blur_radius = intensity * 2
    result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1.0 + intensity * 0.3)
    
    return result

def apply_picasso_style_pil(image, intensity):
    """Apply Picasso-inspired cubist effects using PIL"""
    # Find edges
    result = image.filter(ImageFilter.FIND_EDGES)
    
    # Quantize colors
    quantized = image.quantize(colors=max(4, int(16 - intensity * 8)))
    quantized = quantized.convert('RGB')
    
    # Blend edge detection with quantized colors
    result = Image.blend(quantized, result, 0.3 * intensity)
    
    return result

def apply_monet_style_pil(image, intensity):
    """Apply Monet-inspired impressionist effects using PIL"""
    # Soft blur for impressionist effect
    blur_radius = intensity * 3
    result = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(result)
    result = enhancer.enhance(1.0 + intensity * 0.2)
    
    # Soft color enhancement
    enhancer = ImageEnhance.Color(result)
    result = enhancer.enhance(1.0 + intensity * 0.3)
    
    return result

def apply_kandinsky_style_pil(image, intensity):
    """Apply Kandinsky-inspired abstract effects using PIL"""
    # Strong color enhancement
    enhancer = ImageEnhance.Color(image)
    result = enhancer.enhance(1.0 + intensity * 0.8)
    
    # High contrast
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1.0 + intensity * 0.6)
    
    # Add some blur for abstract feel
    result = result.filter(ImageFilter.GaussianBlur(radius=intensity))
    
    return result

def apply_ukiyo_e_style_pil(image, intensity):
    """Apply Japanese Ukiyo-e woodblock print effects using PIL"""
    # High contrast and posterization
    result = ImageOps.posterize(image, int(8 - intensity * 2))
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1.0 + intensity * 0.5)
    
    # Find edges and overlay
    edges = image.filter(ImageFilter.FIND_EDGES)
    result = Image.blend(result, edges, 0.2 * intensity)
    
    return result

def apply_pop_art_style_pil(image, intensity):
    """Apply Pop Art effects using PIL"""
    # Strong color quantization
    quantized = image.quantize(colors=max(6, int(12 - intensity * 4)))
    result = quantized.convert('RGB')
    
    # Boost colors dramatically
    enhancer = ImageEnhance.Color(result)
    result = enhancer.enhance(1.0 + intensity * 1.2)
    
    # High contrast
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1.0 + intensity * 0.8)
    
    return result

def apply_watercolor_style_pil(image, intensity):
    """Apply watercolor effects using PIL"""
    # Multiple soft blur passes
    result = image
    for _ in range(int(intensity * 3)):
        result = result.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Soft color enhancement
    enhancer = ImageEnhance.Color(result)
    result = enhancer.enhance(1.0 + intensity * 0.4)
    
    # Slight brightness boost
    enhancer = ImageEnhance.Brightness(result)
    result = enhancer.enhance(1.0 + intensity * 0.1)
    
    return result

def preserve_original_colors_pil(original, styled, blend_factor):
    """Preserve original colors by blending with styled image"""
    return Image.blend(styled, original, blend_factor)

if __name__ == "__main__":
    main()
