import cv2
import numpy as np


def to_grayscale(image):
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_filter(gray, kernel_size=3):
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def adaptive_threshold(gray, block_size=21, c_value=10):
    """Each pixel is compared with a Gaussian-weighted mean of its neighbourhood,
    so a shadow on one side of the page does not turn that whole side black."""
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, c_value)


def histogram_equalization(gray):
    return cv2.equalizeHist(gray)


def apply_clahe(gray, clip_limit=2.0, tile_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


# Morphology note: these functions work on the WHITE pixels.
# In a scanned page the background is white and text is black, so:
#   erosion  -> white shrinks  -> text looks thicker
#   dilation -> white grows    -> text looks thinner
#   opening  (erode, then dilate) -> removes small white specks inside text
#   closing  (dilate, then erode) -> removes small black dots on the background

def make_kernel(size):
    return np.ones((size, size), np.uint8)


def erosion(image, size=2):
    return cv2.erode(image, make_kernel(size), iterations=1)


def dilation(image, size=2):
    return cv2.dilate(image, make_kernel(size), iterations=1)


def opening(image, size=2):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, make_kernel(size))


def closing(image, size=2):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, make_kernel(size))


def normalize_illumination(channel):
    """Estimate the page background (text removed by dilation + median blur)
    and divide by it. This flattens shadows and uneven lighting."""
    background = cv2.dilate(channel, make_kernel(7))
    background = cv2.medianBlur(background, 21)
    normalized = cv2.divide(channel, background, scale=255)
    return normalized


# ---------- Scanning modes ----------

def colour_scan(warped):
    return warped.copy()


def grayscale_scan(warped):
    return to_grayscale(warped)


def black_and_white_scan(warped, block_size=21, c_value=10, morph_size=2):
    gray = to_grayscale(warped)
    smooth = gaussian_filter(gray, 3)
    binary = adaptive_threshold(smooth, block_size, c_value)
    if morph_size > 1:
        binary = closing(binary, morph_size)
    return binary


def enhanced_scan(warped, clip_limit=2.0):
    """Colour scan with shadows removed and local contrast boosted."""
    channels = cv2.split(warped)
    flat = cv2.merge([normalize_illumination(ch) for ch in channels])

    # CLAHE only on the lightness channel so colours do not shift
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = apply_clahe(l_channel, clip_limit)
    result = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    return result


def all_techniques(warped, block_size=21, c_value=10, clip_limit=2.0, morph_size=2):
    """Every individual technique, so the user can compare them side by side."""
    gray = to_grayscale(warped)
    binary = adaptive_threshold(gaussian_filter(gray, 3), block_size, c_value)
    return {
        "Grayscale": gray,
        "Gaussian Filter": gaussian_filter(gray, 5),
        "Adaptive Threshold": binary,
        "Histogram Equalization": histogram_equalization(gray),
        "CLAHE": apply_clahe(gray, clip_limit),
        "Erosion (on B&W)": erosion(binary, morph_size),
        "Dilation (on B&W)": dilation(binary, morph_size),
        "Opening (on B&W)": opening(binary, morph_size),
        "Closing (on B&W)": closing(binary, morph_size),
    }
