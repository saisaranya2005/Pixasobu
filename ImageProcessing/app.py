import streamlit as st
import cv2
import numpy as np

# Preprocessing Techniques
def histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def contrast_stretching(img, s_min=0, s_max=255):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r_min = np.min(img_gray)
    r_max = np.max(img_gray)
    stretched = ((img_gray - r_min) / (r_max - r_min)) * (s_max - s_min) + s_min
    return cv2.cvtColor(np.uint8(np.clip(stretched, 0, 255)), cv2.COLOR_GRAY2BGR)

def gamma_correction(img, gamma=1.0):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma_corrected = np.power(img_gray / 255.0, gamma) * 255.0
    return cv2.cvtColor(np.uint8(gamma_corrected), cv2.COLOR_GRAY2BGR)

def log_transformation(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c = 255 / (np.log(1 + np.max(img_gray)))
    epsilon = 1e-5  # Add a small constant to avoid divide-by-zero
    log_transformed = c * np.log(1 + img_gray + epsilon)
    return cv2.cvtColor(np.uint8(log_transformed), cv2.COLOR_GRAY2BGR)

def negative_transformation(img):
    negative = 255 - img
    return negative

def gaussian_blur(img, kernel_size=5):
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred

def sharpening(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def cartoon_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def watercolor_effect(img):
    watercolor = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    return watercolor

def sobel_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    sobel_combined = np.uint8(sobel_combined)
    return cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)

# Define explanations for each enhancement technique
def display_explanations():
    st.title("Enhancement Techniques Explained 🎨")
    techniques = {
        "Histogram Equalization 📊": (
            "Improves the contrast in grayscale images by distributing intensities."
        ),
        "Contrast Stretching 🔍": (
            "Expands the range of intensities in an image to improve contrast."
        ),
        "Gamma Correction 🌟": (
            "Adjusts brightness using a non-linear transformation."
        ),
        "Log Transformation 📈": (
            "Enhances low-intensity values in images with high dynamic ranges."
        ),
        "Negative Transformation 🌌": (
            "Creates a negative of the image."
        ),
        "Gaussian Blur 🌫️": (
            "Reduces image noise and detail using a smoothing filter."
        ),
        "Sharpening ✨": (
            "Enhances edges and details in an image."
        ),
        "Cartoon Effect 🖼️": (
            "Gives images a cartoonish appearance by combining edge detection with smoothing."
        ),
        "Watercolor Effect 🎨": (
            "Applies a painting-like stylization to the image."
        ),
        "Sobel Edge Detection 🖍️": (
            "Highlights edges by computing gradients in the X and Y directions."
        ),
    }
    for technique, description in techniques.items():
        st.subheader(technique)
        st.write(description)
        st.write("---")

# Main app with navigation
st.sidebar.title("Pixasobu 🪐")
page = st.sidebar.radio("Navigate", ["Enhance Image", "Techniques Explained"])

if page == "Enhance Image":
    st.title("Pixasobu 🪐: Ultimate Play with Pixels")
    st.sidebar.title("Enhancement Options")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
        original_image = image.copy()

        enhancement_type = st.sidebar.selectbox(
            "Choose an enhancement technique",
            [
                "Histogram Equalization",
                "Contrast Stretching",
                "Gamma Correction",
                "Log Transformation",
                "Negative Transformation",
                "Gaussian Blur",
                "Sharpening",
                "Cartoon Effect",
                "Watercolor Effect",
                "Sobel Edge Detection"
            ]
        )

        if enhancement_type == "Histogram Equalization":
            result = histogram_equalization(original_image)
        elif enhancement_type == "Contrast Stretching":
            result = contrast_stretching(original_image)
        elif enhancement_type == "Gamma Correction":
            gamma = st.sidebar.slider("Gamma Value", 0.1, 5.0, 1.0)
            result = gamma_correction(original_image, gamma)
        elif enhancement_type == "Log Transformation":
            result = log_transformation(original_image)
        elif enhancement_type == "Negative Transformation":
            result = negative_transformation(original_image)
        elif enhancement_type == "Gaussian Blur":
            kernel_size = st.sidebar.slider("Kernel Size (odd only)", 3, 21, 5, step=2)
            result = gaussian_blur(original_image, kernel_size)
        elif enhancement_type == "Sharpening":
            result = sharpening(original_image)
        elif enhancement_type == "Cartoon Effect":
            result = cartoon_effect(original_image)
        elif enhancement_type == "Watercolor Effect":
            result = watercolor_effect(original_image)
        elif enhancement_type == "Sobel Edge Detection":
            result = sobel_edge_detection(original_image)

        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Enhanced Image", use_container_width=True)

        result_download = cv2.imencode('.jpg', result)[1].tobytes()
        st.download_button(
            label="Download Enhanced Image",
            data=result_download,
            file_name="enhanced_image.jpg",
            mime="image/jpeg")

elif page == "Techniques Explained":
    display_explanations()
