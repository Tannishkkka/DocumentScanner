import io

import cv2
from PIL import Image

A4_WIDTH_INCHES = 8.27


def to_image_bytes(image, file_format):
    """Encode an OpenCV image as JPG or PNG bytes for downloading."""
    extension = ".jpg" if file_format == "JPG" else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension == ".jpg" else []
    success, buffer = cv2.imencode(extension, image, params)
    if not success:
        raise ValueError(f"Could not encode image as {file_format}.")
    return buffer.tobytes()


def to_pdf_bytes(image):
    """Save the image as a one-page PDF. The page size is derived from the image's
    own pixel size, so the aspect ratio is kept exactly (no stretching)."""
    if image.ndim == 2:
        pil_image = Image.fromarray(image).convert("L")
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Choose a DPI so that the page width is A4 width; height follows the ratio
    dpi = pil_image.width / A4_WIDTH_INCHES
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PDF", resolution=dpi)
    return buffer.getvalue()
