import cv2
import numpy as np

MIN_AREA_RATIO = 0.15   # document must cover at least 15% of the photo
MAX_AREA_RATIO = 0.98   # ignore contours that are just the photo border


def remove_text_details(gray):
    """Morphological closing with a big kernel wipes out dark text on the page,
    so that Canny mostly finds the page outline instead of letters."""
    kernel = np.ones((9, 9), np.uint8)
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)


def find_edges(image, low, high):
    edges = cv2.Canny(image, low, high)
    # Dilation joins small gaps in the outline so the contour is closed
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def auto_canny_thresholds(image, sigma=0.33):
    """Choose Canny thresholds around the median brightness of the image."""
    median = np.median(image)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    return low, high


def find_quadrilateral(binary_image, image_area):
    """Return the 4 corners of the largest convex 4-sided contour, or None."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MAX_AREA_RATIO * image_area:
            continue
        if area < MIN_AREA_RATIO * image_area:
            break  # list is sorted, so the rest are even smaller
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def detect_document(prep):
    """Try a few edge/threshold settings and return the first good 4-corner contour.

    prep is the dictionary returned by prepare_for_detection().
    Returns (corners in ORIGINAL image coordinates or None, edge image, method name).
    """
    gray = prep["denoised"]
    cleaned = remove_text_details(gray)
    image_area = gray.shape[0] * gray.shape[1]

    attempts = []
    auto_low, auto_high = auto_canny_thresholds(cleaned)
    attempts.append(("Canny (auto thresholds)", find_edges(cleaned, auto_low, auto_high)))
    attempts.append(("Canny (75, 200)", find_edges(cleaned, 75, 200)))
    attempts.append(("Canny (30, 100)", find_edges(cleaned, 30, 100)))

    # Backup: Otsu threshold separates a bright page from a darker background
    _, otsu = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    attempts.append(("Otsu threshold", otsu))

    for method, binary in attempts:
        corners = find_quadrilateral(binary, image_area)
        if corners is not None:
            return corners * prep["ratio"], binary, method

    # Nothing worked: return the first edge map so the user can see why
    return None, attempts[0][1], "None"


def refine_corners(image, corners, ratio):
    """Detection runs on a small copy, so each corner can be a few pixels off in the
    full image. cv2.cornerSubPix moves each point to the exact corner location
    using image gradients in a small window around it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    window = int(min(15, max(3, ratio * 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    points = corners.reshape(-1, 1, 2).astype(np.float32).copy()
    refined = cv2.cornerSubPix(gray, points, (window, window), (-1, -1), criteria).reshape(4, 2)

    # If a point jumped far away (e.g. pulled by nearby text), keep the original
    moved = np.linalg.norm(refined - corners, axis=1)
    refined[moved > window] = corners[moved > window]
    return refined


def draw_boundary(image, corners, color=(0, 200, 0)):
    output = image.copy()
    thickness = max(2, image.shape[1] // 250)
    pts = corners.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(output, [pts], True, color, thickness)
    for x, y in corners:
        cv2.circle(output, (int(x), int(y)), thickness * 3, (0, 0, 255), -1)
    return output
