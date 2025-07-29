# 🎨 Art Style Transfer Studio

Transform your photos into artistic masterpieces inspired by famous artists using advanced computer vision techniques.

## ✨ Features

- **8 Artistic Styles**: Van Gogh, Picasso, Monet, Kandinsky, Japanese Ukiyo-e, Pop Art, Watercolor
- **Adjustable Intensity**: Control style strength from subtle (0.1) to dramatic (2.0)
- **Color Preservation**: Option to maintain original colors while applying artistic effects
- **Multiple Resolutions**: Choose output quality from 256px to 1024px
- **EXIF Handling**: Automatic image orientation correction for mobile photos
- **Download Support**: Save your artwork as high-quality PNG files

## 🎭 Available Art Styles

| Style | Description | Best For |
|-------|-------------|----------|
| **Van Gogh** | Swirling brushstrokes with vibrant colors | Landscapes, portraits |
| **Picasso** | Geometric cubist forms with edge detection | Faces, structured subjects |
| **Monet** | Soft impressionist blur with enhanced light | Nature, outdoor scenes |
| **Kandinsky** | Abstract geometric distortions | Creative artistic expression |
| **Japanese Ukiyo-e** | High contrast woodblock print style | Traditional art aesthetic |
| **Pop Art** | Bright colors with high contrast | Modern, vibrant subjects |
| **Watercolor** | Soft, organic textures with paper effects | Gentle, artistic renditions |

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/art-style-transfer-studio.git
   cd art-style-transfer-studio
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_for_deployment.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Streamlit Cloud Deployment

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy using your forked repository
4. The app will automatically install dependencies and start

## 📱 Usage Instructions

1. **Upload Image**: Click "Choose an image..." and select your photo
2. **Select Style**: Choose from 8 artistic styles in the sidebar
3. **Adjust Settings**: 
   - Style intensity (0.1-2.0)
   - Enable color preservation if desired
   - Select output resolution
4. **Generate Art**: Click "🚀 Generate Art" button
5. **Download**: Save your styled image using the download button

## 💡 Tips for Best Results

### Text Preservation
- **Best for text**: Pop Art, Japanese Ukiyo-e (maintain sharp edges)
- **Moderate impact**: Van Gogh, Picasso at low intensity (0.3-0.7)
- **Softens text**: Watercolor, Monet (blur effects)

### Style Intensity Guide
- **0.1-0.5**: Subtle artistic enhancement
- **0.6-1.0**: Balanced artistic transformation
- **1.1-1.5**: Strong artistic effect
- **1.6-2.0**: Maximum artistic impact

### Image Quality
- Use high-resolution source images for best results
- JPEG and PNG formats are supported
- Mobile photos are automatically oriented correctly

## 🛠️ Technical Architecture

### Core Components
- **Frontend**: Streamlit web interface with responsive design
- **Image Processing**: OpenCV and PIL for robust image handling
- **Style Transfer**: Traditional computer vision techniques for reliability
- **Color Management**: RGB pipeline with proper EXIF orientation handling

### Style Algorithms
- **Van Gogh**: Color enhancement with canvas texture and artistic blur
- **Picasso**: Edge detection with k-means color quantization for cubist effects
- **Monet**: Gaussian blur with HSV color space enhancement
- **Kandinsky**: Geometric wave distortions with color boosting
- **Ukiyo-e**: High contrast quantization with edge overlays
- **Pop Art**: Bright color clustering with contrast enhancement
- **Watercolor**: Bilateral filtering with organic paper textures

## 📋 System Requirements

- Python 3.8 or higher
- 2GB RAM minimum (4GB recommended)
- Modern web browser
- Internet connection for initial setup

## 🔧 Dependencies

- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and image processing
- **Pillow**: Image manipulation and format handling
- **NumPy**: Numerical operations on image arrays

## 🎯 Performance Optimization

- **Caching**: Streamlit decorators for efficient processing
- **Memory Management**: Optimized array operations
- **Resolution Control**: User-selectable output sizes
- **Error Handling**: Graceful fallbacks for edge cases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by traditional art techniques of Van Gogh, Picasso, Monet, and Kandinsky
- Built with modern computer vision and web technologies
- Optimized for both desktop and mobile usage

## 📸 Screenshots

Upload any photo and transform it into artistic masterpieces with adjustable intensity and style preservation options.

---

**Made with ❤️ using Streamlit and OpenCV**