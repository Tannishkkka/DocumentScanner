import cv2
import numpy as np


def order_points(points):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    Points are sorted by their angle around the centre, which gives a clockwise
    order (image y-axis points down). Then the list is rotated so that the point
    with the smallest x + y (closest to the image origin) comes first.
    """
    points = np.array(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    clockwise = points[np.argsort(angles)]

    start = np.argmin(clockwise.sum(axis=1))
    return np.roll(clockwise, -start, axis=0)


def distance(p1, p2):
    return float(np.sqrt(((p1 - p2) ** 2).sum()))


def four_point_transform(image, corners):
    """Warp the region inside the 4 corners into a flat, top-down rectangle."""
    tl, tr, br, bl = order_points(corners)

    # Output size = the longer of the two opposite edges
    width = int(max(distance(tl, tr), distance(bl, br)))
    height = int(max(distance(tl, bl), distance(tr, br)))
    if width < 10 or height < 10:
        raise ValueError("Selected document region is too small to warp.")

    source = np.array([tl, tr, br, bl], dtype=np.float32)
    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix
