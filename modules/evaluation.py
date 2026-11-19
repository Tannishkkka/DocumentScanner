import cv2
import numpy as np


def prepare_pair(result, reference):
    """Convert both images to grayscale float and resize result to the reference size."""
    if result.ndim == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    height, width = reference.shape[:2]
    result = cv2.resize(result, (width, height), interpolation=cv2.INTER_AREA)
    return result.astype(np.float64), reference.astype(np.float64)


def mse(result, reference):
    a, b = prepare_pair(result, reference)
    return float(np.mean((a - b) ** 2))


def psnr(result, reference):
    error = mse(result, reference)
    if error == 0:
        return float("inf")
    return float(10 * np.log10((255.0 ** 2) / error))


def ssim(result, reference):
    """SSIM (Wang et al., 2004) with an 11x11 Gaussian window, sigma 1.5."""
    a, b = prepare_pair(result, reference)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    def blur(img):
        return cv2.GaussianBlur(img, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov_ab = blur(a * b) - mu_a * mu_b

    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)) / \
               ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(ssim_map.mean())


def evaluate(result, reference):
    return {"MSE": mse(result, reference), "PSNR (dB)": psnr(result, reference),
            "SSIM": ssim(result, reference)}
