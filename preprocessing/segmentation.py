"""
Dental Crown Image Segmentation Module.

Phase 1 integrates Segment Anything Model (SAM) as the primary segmentation engine,
while preserving the baseline Otsu thresholding method for comparative benchmarks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

from .sam_segmentation import DentalSAMSegmentor, segment_dental_sample


# Global singleton segmentor cache to avoid reloading weights repeatedly in memory
_CACHED_SAM_SEGMENTOR: Optional[DentalSAMSegmentor] = None


def get_sam_segmentor(
    checkpoint_path: Optional[str | Path] = None,
    model_type: str = "vit_b",
    device: Optional[str] = None,
) -> DentalSAMSegmentor:
    """
    Returns a cached DentalSAMSegmentor instance or initializes a new one.
    """
    global _CACHED_SAM_SEGMENTOR
    if _CACHED_SAM_SEGMENTOR is None:
        _CACHED_SAM_SEGMENTOR = DentalSAMSegmentor(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            device=device,
        )
    return _CACHED_SAM_SEGMENTOR


def apply_sam_segmentation(
    image_np: np.ndarray,
    checkpoint_path: Optional[str | Path] = None,
    model_type: str = "vit_b",
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 1: Applies SAM (Segment Anything Model) to segment dental crowns/teeth from background.
    
    Returns:
        tuple[np.ndarray, np.ndarray]:
            - segmented_image: shape (H, W, 3), background 0.
            - mask: shape (H, W), binary uint8 (0=background, 255=foreground).
    """
    segmentor = get_sam_segmentor(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
    )
    segmented, mask, _ = segmentor.segment_image(image_np)
    return segmented, mask


def apply_otsu_segmentation(image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline / Comparison Stage 1: Applies Otsu thresholding to segment dental crown from background.
    Preserved for comparative analysis.
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


def apply_segmentation(
    image_np: np.ndarray,
    method: str = "sam",
    checkpoint_path: Optional[str | Path] = None,
    model_type: str = "vit_b",
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Active entrypoint for Stage 1 Segmentation.
    Defaults to 'sam', with 'otsu' available for baseline comparison.
    """
    if method == "sam":
        return apply_sam_segmentation(
            image_np, checkpoint_path=checkpoint_path, model_type=model_type, device=device
        )
    elif method == "otsu":
        return apply_otsu_segmentation(image_np)
    else:
        raise ValueError(f"Unknown segmentation method '{method}'. Choose 'sam' or 'otsu'.")


def augment_dental_data(image_np: np.ndarray) -> List[np.ndarray]:
    """
    Stage 1: Augmentor / Imgaug data amplification for dental crown views.
    """
    augmented = [image_np]
    bright = np.clip(image_np.astype(np.float32) * 1.05 + 10, 0, 255).astype(np.uint8)
    dim = np.clip(image_np.astype(np.float32) * 0.95 - 5, 0, 255).astype(np.uint8)
    augmented.extend([bright, dim])
    return augmented
