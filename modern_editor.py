import io
from dataclasses import dataclass

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage
from skimage import filters


FILTER_OPTIONS = [
    "None",
    "Blur",
    "Gaussian Blur",
    "Sharpen",
    "Edge Enhance",
    "Emboss",
    "Find Edges",
    "Smooth",
    "Detail",
    "Contour",
]

EFFECT_OPTIONS = [
    "None",
    "Grayscale",
    "Sepia",
    "Invert",
    "Solarize",
    "Posterize",
    "Equalize",
    "Auto Contrast",
]

EDGE_OPTIONS = ["Sobel", "Canny", "Laplacian", "Prewitt", "Roberts"]
DENOISE_OPTIONS = ["Gaussian", "Median", "Bilateral", "NL-Means"]
MORPH_OPTIONS = ["Erosion", "Dilation", "Opening", "Closing"]
STYLE_OPTIONS = ["Warm", "Cool", "Vintage", "Cinematic", "Pop Art"]


@dataclass(frozen=True)
class BasicSettings:
    brightness: int
    contrast: int
    saturation: int
    sharpness: int


def configure_page() -> None:
    st.set_page_config(page_title="AI Image Editor Pro", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #14181f 0%, #0f1218 100%);
            color: #f3f5f7;
        }
        [data-testid="stSidebar"], .stTabs [data-baseweb="tab-panel"] {
            background: rgba(24, 29, 38, 0.92);
        }
        .editor-card {
            background: rgba(28, 34, 44, 0.95);
            border: 1px solid #323846;
            border-radius: 18px;
            padding: 1rem 1.2rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
        }
        .info-text {
            color: #aab4c3;
            font-size: 0.95rem;
            margin-top: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "current_image": None,
        "original_image": None,
        "image_name": None,
        "status_message": "Ready to edit images!",
        "last_basic_settings": BasicSettings(0, 0, 0, 0),
        "last_rotation": 0,
        "uploader_signature": None,
        "brightness": 0,
        "contrast": 0,
        "saturation": 0,
        "sharpness": 0,
        "rotation": 0,
        "filter_name": FILTER_OPTIONS[0],
        "effect_name": EFFECT_OPTIONS[0],
        "edge_method": EDGE_OPTIONS[0],
        "denoise_method": DENOISE_OPTIONS[0],
        "morph_operation": MORPH_OPTIONS[0],
        "sr_scale": 2,
        "style_name": STYLE_OPTIONS[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_status(message: str) -> None:
    st.session_state.status_message = message


def reset_controls() -> None:
    st.session_state.brightness = 0
    st.session_state.contrast = 0
    st.session_state.saturation = 0
    st.session_state.sharpness = 0
    st.session_state.rotation = 0
    st.session_state.last_basic_settings = BasicSettings(0, 0, 0, 0)
    st.session_state.last_rotation = 0


def load_uploaded_image(uploaded_file) -> None:
    if uploaded_file is None:
        return

    signature = (uploaded_file.name, uploaded_file.size)
    if signature == st.session_state.uploader_signature:
        return

    image = Image.open(uploaded_file)
    st.session_state.original_image = image.copy()
    st.session_state.current_image = image.copy()
    st.session_state.image_name = uploaded_file.name
    st.session_state.uploader_signature = signature
    reset_controls()
    set_status(f"Loaded: {uploaded_file.name}")


def get_current_image() -> Image.Image | None:
    return st.session_state.current_image


def get_original_image() -> Image.Image | None:
    return st.session_state.original_image


def require_image() -> bool:
    if get_current_image() is None:
        st.warning("Please load an image first!")
        return False
    return True


def pil_to_download_bytes(image: Image.Image) -> tuple[bytes, str, str]:
    output = io.BytesIO()
    if image.mode == "RGBA":
        image.save(output, format="PNG")
        return output.getvalue(), "edited_image.png", "image/png"

    image.save(output, format="PNG")
    return output.getvalue(), "edited_image.png", "image/png"


def update_info_text(image: Image.Image | None) -> str:
    if image is None:
        return "No image loaded"
    width, height = image.size
    return f"Size: {width}x{height} | Mode: {image.mode}"


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    max_value = float(array.max()) if array.size else 0.0
    if max_value <= 0:
        return np.zeros_like(array, dtype=np.uint8)
    return np.clip((array / max_value) * 255, 0, 255).astype(np.uint8)


def rgb_image(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def invert_image(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        red, green, blue, alpha = image.split()
        rgb = Image.merge("RGB", (red, green, blue))
        inverted = ImageOps.invert(rgb)
        return Image.merge("RGBA", (*inverted.split(), alpha))
    return ImageOps.invert(rgb_image(image))


def apply_basic_adjustments_to_original(settings: BasicSettings) -> None:
    original = get_original_image()
    if original is None:
        return

    image = original.copy()
    image = ImageEnhance.Brightness(image).enhance(1.0 + settings.brightness / 100.0)
    image = ImageEnhance.Contrast(image).enhance(1.0 + settings.contrast / 100.0)
    image = ImageEnhance.Color(image).enhance(1.0 + settings.saturation / 100.0)
    image = ImageEnhance.Sharpness(image).enhance(1.0 + settings.sharpness / 100.0)
    st.session_state.current_image = image


def apply_rotation_to_original(angle: int) -> None:
    original = get_original_image()
    if original is None:
        return
    st.session_state.current_image = original.copy().rotate(angle, expand=True)


def apply_slider_changes() -> None:
    original = get_original_image()
    if original is None:
        return

    current_basic = BasicSettings(
        st.session_state.brightness,
        st.session_state.contrast,
        st.session_state.saturation,
        st.session_state.sharpness,
    )
    if current_basic != st.session_state.last_basic_settings:
        apply_basic_adjustments_to_original(current_basic)
        st.session_state.last_basic_settings = current_basic

    current_rotation = int(st.session_state.rotation)
    if current_rotation != st.session_state.last_rotation:
        apply_rotation_to_original(current_rotation)
        st.session_state.last_rotation = current_rotation


def apply_filter(filter_name: str) -> None:
    if not require_image():
        return

    image = get_current_image()
    try:
        if filter_name == "Blur":
            image = image.filter(ImageFilter.BLUR)
        elif filter_name == "Gaussian Blur":
            image = image.filter(ImageFilter.GaussianBlur(2))
        elif filter_name == "Sharpen":
            image = image.filter(ImageFilter.SHARPEN)
        elif filter_name == "Edge Enhance":
            image = image.filter(ImageFilter.EDGE_ENHANCE)
        elif filter_name == "Emboss":
            image = image.filter(ImageFilter.EMBOSS)
        elif filter_name == "Find Edges":
            image = image.filter(ImageFilter.FIND_EDGES)
        elif filter_name == "Smooth":
            image = image.filter(ImageFilter.SMOOTH)
        elif filter_name == "Detail":
            image = image.filter(ImageFilter.DETAIL)
        elif filter_name == "Contour":
            image = image.filter(ImageFilter.CONTOUR)

        st.session_state.current_image = image
        set_status(f"Applied filter: {filter_name}")
    except Exception as exc:
        st.error(f"Failed to apply filter: {exc}")


def apply_sepia(image: Image.Image) -> Image.Image:
    working = rgb_image(image)
    width, height = working.size
    pixels = working.load()

    for py in range(height):
        for px in range(width):
            red, green, blue = working.getpixel((px, py))
            tr = int(0.393 * red + 0.769 * green + 0.189 * blue)
            tg = int(0.349 * red + 0.686 * green + 0.168 * blue)
            tb = int(0.272 * red + 0.534 * green + 0.131 * blue)
            pixels[px, py] = (min(255, tr), min(255, tg), min(255, tb))

    return working


def apply_color_effect(effect_name: str) -> None:
    if not require_image():
        return

    image = get_current_image()
    try:
        if effect_name == "Grayscale":
            image = ImageOps.grayscale(image)
        elif effect_name == "Sepia":
            image = apply_sepia(image)
        elif effect_name == "Invert":
            image = invert_image(image)
        elif effect_name == "Solarize":
            image = ImageOps.solarize(rgb_image(image), 128)
        elif effect_name == "Posterize":
            image = ImageOps.posterize(rgb_image(image), 4)
        elif effect_name == "Equalize":
            image = ImageOps.equalize(rgb_image(image))
        elif effect_name == "Auto Contrast":
            image = ImageOps.autocontrast(rgb_image(image))

        st.session_state.current_image = image
        set_status(f"Applied effect: {effect_name}")
    except Exception as exc:
        st.error(f"Failed to apply effect: {exc}")


def apply_edge_detection(method: str) -> None:
    if not require_image():
        return

    try:
        img_array = np.array(get_current_image().convert("L"))
        if method == "Sobel":
            sobelx = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=5)
            edges = np.sqrt(sobelx ** 2 + sobely ** 2)
        elif method == "Canny":
            edges = cv2.Canny(img_array, 100, 200)
        elif method == "Laplacian":
            edges = cv2.Laplacian(img_array, cv2.CV_64F)
        elif method == "Prewitt":
            kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            prewittx = ndimage.convolve(img_array, kernelx)
            prewitty = ndimage.convolve(img_array, kernely)
            edges = np.sqrt(prewittx ** 2 + prewitty ** 2)
        else:
            edges = filters.roberts(img_array)

        st.session_state.current_image = Image.fromarray(normalize_to_uint8(edges))
        set_status(f"Applied edge detection: {method}")
    except Exception as exc:
        st.error(f"Failed to apply edge detection: {exc}")


def apply_denoising(method: str) -> None:
    if not require_image():
        return

    try:
        image = get_current_image()
        img_array = np.array(image)
        color_image = len(img_array.shape) == 3 and img_array.shape[2] >= 3

        if method == "Gaussian":
            denoised = cv2.GaussianBlur(img_array, (5, 5), 0)
        elif method == "Median":
            denoised = cv2.medianBlur(img_array, 5)
        elif method == "Bilateral":
            denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
        elif color_image:
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)

        st.session_state.current_image = Image.fromarray(denoised)
        set_status(f"Applied denoising: {method}")
    except Exception as exc:
        st.error(f"Failed to apply denoising: {exc}")


def apply_morphological(operation: str) -> None:
    if not require_image():
        return

    try:
        img_array = np.array(get_current_image().convert("L"))
        kernel = np.ones((5, 5), np.uint8)

        if operation == "Erosion":
            result = cv2.erode(img_array, kernel, iterations=1)
        elif operation == "Dilation":
            result = cv2.dilate(img_array, kernel, iterations=1)
        elif operation == "Opening":
            result = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
        else:
            result = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)

        st.session_state.current_image = Image.fromarray(result)
        set_status(f"Applied morphological operation: {operation}")
    except Exception as exc:
        st.error(f"Failed to apply morphological operation: {exc}")


def auto_enhance() -> None:
    if not require_image():
        return

    try:
        img_array = np.array(get_current_image())
        if len(img_array.shape) == 3:
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            lightness, channel_a, channel_b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lightness = clahe.apply(lightness)
            lab = cv2.merge([lightness, channel_a, channel_b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        else:
            enhanced = cv2.equalizeHist(img_array)
            enhanced = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
            enhanced = cv2.addWeighted(img_array, 1.5, enhanced, -0.5, 0)

        st.session_state.current_image = Image.fromarray(enhanced)
        set_status("Applied advanced auto-enhancement")
    except Exception as exc:
        st.error(f"Failed to apply auto-enhance: {exc}")


def super_resolution(scale: int) -> None:
    if not require_image():
        return

    try:
        img_array = np.array(get_current_image())
        new_width = int(img_array.shape[1] * scale)
        new_height = int(img_array.shape[0] * scale)
        interpolation = cv2.INTER_LANCZOS4 if len(img_array.shape) == 3 else cv2.INTER_CUBIC
        upscaled = cv2.resize(img_array, (new_width, new_height), interpolation=interpolation)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        upscaled = cv2.filter2D(upscaled, -1, kernel)

        st.session_state.current_image = Image.fromarray(upscaled)
        set_status(f"Upscaled image by {scale}x")
    except Exception as exc:
        st.error(f"Failed to apply super resolution: {exc}")


def apply_vintage_style(img_array: np.ndarray) -> np.ndarray:
    working = img_array.astype(np.float32)
    transformed = np.empty_like(working)
    transformed[:, :, 0] = np.clip(0.393 * working[:, :, 0] + 0.769 * working[:, :, 1] + 0.189 * working[:, :, 2], 0, 255)
    transformed[:, :, 1] = np.clip(0.349 * working[:, :, 0] + 0.686 * working[:, :, 1] + 0.168 * working[:, :, 2], 0, 255)
    transformed[:, :, 2] = np.clip(0.272 * working[:, :, 0] + 0.534 * working[:, :, 1] + 0.131 * working[:, :, 2], 0, 255)
    return transformed.astype(np.uint8)


def style_transfer(style: str) -> None:
    if not require_image():
        return

    try:
        img_array = np.array(rgb_image(get_current_image()))

        if style == "Warm":
            img_array = img_array.astype(np.float32)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.2, 0, 255)
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.1, 0, 255)
        elif style == "Cool":
            img_array = img_array.astype(np.float32)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.2, 0, 255)
        elif style == "Vintage":
            img_array = apply_vintage_style(img_array)
        elif style == "Cinematic":
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            lightness, channel_a, channel_b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0)
            lightness = clahe.apply(lightness)
            channel_b = np.clip(channel_b.astype(np.int16) + 10, 0, 255).astype(np.uint8)
            img_array = cv2.merge([lightness, channel_a, channel_b])
            img_array = cv2.cvtColor(img_array, cv2.COLOR_LAB2RGB)
        elif style == "Pop Art":
            image = Image.fromarray(img_array)
            image = ImageOps.posterize(image, 4)
            image = ImageEnhance.Color(image).enhance(1.5)
            img_array = np.array(image)

        st.session_state.current_image = Image.fromarray(img_array.astype(np.uint8))
        set_status(f"Applied {style} style")
    except Exception as exc:
        st.error(f"Failed to apply style transfer: {exc}")


def blur_faces() -> None:
    if not require_image():
        return

    try:
        img_array = np.array(rgb_image(get_current_image()))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

        faces1 = face_cascade.detectMultiScale(gray, 1.1, 4)
        faces2 = profile_cascade.detectMultiScale(gray, 1.1, 4)
        all_faces = list(faces1) + list(faces2)

        if len(all_faces) == 0:
            st.info("No faces detected in the image")
            return

        for x, y, width, height in all_faces:
            x = max(0, x - 10)
            y = max(0, y - 10)
            width = min(width + 20, img_array.shape[1] - x)
            height = min(height + 20, img_array.shape[0] - y)
            face_roi = img_array[y : y + height, x : x + width]
            blur_strength = max(25, min(99, width // 4))
            if blur_strength % 2 == 0:
                blur_strength += 1
            img_array[y : y + height, x : x + width] = cv2.GaussianBlur(
                face_roi, (blur_strength, blur_strength), 0
            )

        st.session_state.current_image = Image.fromarray(img_array)
        set_status(f"Blurred {len(all_faces)} faces")
    except Exception as exc:
        st.error(f"Failed to blur faces: {exc}")


def cartoon_effect() -> None:
    if not require_image():
        return

    try:
        img_array = np.array(rgb_image(get_current_image()))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        filtered = cv2.bilateralFilter(img_array, 9, 75, 75)
        edges = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2,
        )
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        cartoon = cv2.bitwise_and(filtered, edges)

        st.session_state.current_image = Image.fromarray(cartoon)
        set_status("Applied cartoon effect")
    except Exception as exc:
        st.error(f"Failed to apply cartoon effect: {exc}")


def remove_background() -> None:
    if not require_image():
        return

    try:
        img_array = np.array(rgb_image(get_current_image()))
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([180, 255, 200])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.bitwise_not(mask)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        rgba = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask

        st.session_state.current_image = Image.fromarray(rgba)
        set_status("Applied background removal (simple)")
    except Exception as exc:
        st.error(f"Failed to remove background: {exc}")


def render_header() -> None:
    st.title("AI Image Editor Pro")
    st.caption("Web version of the original editor with the same editing tools.")


def render_toolbar() -> None:
    uploaded_file = st.file_uploader(
        "Open Image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        label_visibility="collapsed",
    )
    load_uploaded_image(uploaded_file)

    toolbar_columns = st.columns([1, 1, 1])
    with toolbar_columns[0]:
        st.write("Upload an image to start editing.")
    with toolbar_columns[1]:
        image = get_current_image()
        if image is not None:
            data, filename, mime = pil_to_download_bytes(image)
            st.download_button("Save Image", data=data, file_name=filename, mime=mime, use_container_width=True)
        else:
            st.button("Save Image", disabled=True, use_container_width=True)
    with toolbar_columns[2]:
        if st.button("Reset", use_container_width=True, disabled=get_original_image() is None):
            st.session_state.current_image = get_original_image().copy()
            reset_controls()
            set_status("Image reset to original")
            st.rerun()


def render_preview() -> None:
    image = get_current_image()
    st.markdown('<div class="editor-card">', unsafe_allow_html=True)
    if image is None:
        st.info("No image loaded. Upload an image to start editing.")
    else:
        st.image(image, use_container_width=True)
    st.markdown(
        f'<div class="info-text">{update_info_text(image)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_basic_tab(tab) -> None:
    with tab:
        st.slider("Brightness", -100, 100, key="brightness")
        st.slider("Contrast", -100, 100, key="contrast")
        st.slider("Saturation", -100, 100, key="saturation")
        st.slider("Sharpness", -100, 100, key="sharpness")
        st.slider("Rotation", -180, 180, key="rotation")


def render_filters_tab(tab) -> None:
    with tab:
        st.selectbox("Image Filters", FILTER_OPTIONS, key="filter_name")
        if st.button("Apply Filter", use_container_width=True):
            apply_filter(st.session_state.filter_name)

        st.selectbox("Color Effects", EFFECT_OPTIONS, key="effect_name")
        if st.button("Apply Effect", use_container_width=True):
            apply_color_effect(st.session_state.effect_name)


def render_advanced_tab(tab) -> None:
    with tab:
        st.selectbox("Edge Detection", EDGE_OPTIONS, key="edge_method")
        if st.button("Detect Edges", use_container_width=True):
            apply_edge_detection(st.session_state.edge_method)

        st.selectbox("Noise Reduction", DENOISE_OPTIONS, key="denoise_method")
        if st.button("Remove Noise", use_container_width=True):
            apply_denoising(st.session_state.denoise_method)

        st.selectbox("Morphological Operations", MORPH_OPTIONS, key="morph_operation")
        if st.button("Apply Operation", use_container_width=True):
            apply_morphological(st.session_state.morph_operation)


def render_ai_tab(tab) -> None:
    with tab:
        if st.button("Auto Enhance", use_container_width=True):
            auto_enhance()

        st.slider("Super Resolution Scale", 2, 4, key="sr_scale")
        if st.button("Super Resolution", use_container_width=True):
            super_resolution(st.session_state.sr_scale)

        st.selectbox("Style Transfer", STYLE_OPTIONS, key="style_name")
        if st.button("Apply Style", use_container_width=True):
            style_transfer(st.session_state.style_name)

        if st.button("Blur Faces", use_container_width=True):
            blur_faces()

        if st.button("Cartoon Effect", use_container_width=True):
            cartoon_effect()

        if st.button("Remove Background", use_container_width=True):
            remove_background()


def render_controls() -> None:
    st.markdown('<div class="editor-card">', unsafe_allow_html=True)
    tabs = st.tabs(["Basic", "Filters", "Advanced", "AI Tools"])
    render_basic_tab(tabs[0])
    render_filters_tab(tabs[1])
    render_advanced_tab(tabs[2])
    render_ai_tab(tabs[3])
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    configure_page()
    init_state()
    render_header()
    render_toolbar()
    apply_slider_changes()

    preview_column, controls_column = st.columns([7, 3], gap="large")
    with preview_column:
        render_preview()
    with controls_column:
        render_controls()

    st.caption(st.session_state.status_message)


if __name__ == "__main__":
    main()
