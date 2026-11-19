import cv2
import numpy as np

DETECTION_HEIGHT = 500


def load_image_from_bytes(file_bytes):
    """Decode uploaded file bytes into an OpenCV BGR image. Returns None if invalid."""
    if not file_bytes:
        return None
    data = np.frombuffer(file_bytes, dtype=np.uint8)
    try:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    return image


def resize_image(image, target_height=DETECTION_HEIGHT):
    """Shrink the image to a fixed height for faster detection.
    Returns the resized image and the ratio needed to scale points back."""
    height, width = image.shape[:2]
    if height <= target_height:
        return image.copy(), 1.0
    ratio = height / target_height
    new_width = int(width / ratio)
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, ratio


def preprocess_image(image):
    """Grayscale -> Gaussian blur -> median blur (noise reduction)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Median blur removes isolated noisy pixels (salt-and-pepper) while keeping edges sharp
    denoised = cv2.medianBlur(blurred, 5)
    return gray, blurred, denoised


def prepare_for_detection(image):
    resized, ratio = resize_image(image)
    gray, blurred, denoised = preprocess_image(resized)
    return {
        "resized": resized,
        "ratio": ratio,
        "gray": gray,
        "blurred": blurred,
        "denoised": denoised,
    }
