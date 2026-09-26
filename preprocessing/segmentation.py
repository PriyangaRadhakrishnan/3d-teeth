import numpy as np
from PIL import Image

def apply_otsu_segmentation(image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 1: Applies Otsu thresholding to segment the dental crown/tooth region from background
    and remove outliers/noise without external C++ opencv dependencies.
    """
    if image_np.ndim == 3:
        gray = np.mean(image_np, axis=2).astype(np.uint8)
    else:
        gray = image_np.astype(np.uint8)

    # Otsu thresholding algorithm implementation in pure NumPy
    hist, bin_edges = np.histogram(gray, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / np.maximum(weight2, 1)

    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance)
    threshold = bin_centers[:-1][idx]

    mask = (gray > threshold).astype(np.uint8) * 255
    if image_np.ndim == 3:
        segmented = image_np * (mask[:, :, None] // 255)
    else:
        segmented = image_np * (mask // 255)

    return segmented, mask

def augment_dental_data(image_np: np.ndarray) -> list[np.ndarray]:
    """
    Stage 1: Augmentor / Imgaug data amplification for dental crown views.
    """
    augmented = [image_np]
    bright = np.clip(image_np.astype(np.float32) * 1.05 + 10, 0, 255).astype(np.uint8)
    dim = np.clip(image_np.astype(np.float32) * 0.95 - 5, 0, 255).astype(np.uint8)
    augmented.extend([bright, dim])
    return augmented
