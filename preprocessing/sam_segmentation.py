"""
Phase 1: Segment Anything Model (SAM) Based Dental Segmentation.

This module provides a legitimate, fully reproducible SAM-based segmentation
pipeline specifically engineered for 5-view intraoral dental photographs.
It produces binary uint8 masks (0=background, 255=foreground) matching the
exact format and dimensions expected by Aarav's downstream reconstruction pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage as ndi

from segment_anything import sam_model_registry, SamPredictor


DEFAULT_CHECKPOINT = Path("checkpoints/sam_vit_b_01ec64.pth")
DEFAULT_MODEL_TYPE = "vit_b"


class DentalSAMSegmentor:
    """
    SAM-based dental segmentation engine for intraoral dental photographs.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        model_type: str = DEFAULT_MODEL_TYPE,
        device: Optional[str] = None,
        min_area_fraction: float = 0.05,
        max_area_fraction: float = 0.85,
    ) -> None:
        self.model_type = model_type
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction

        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT
        self.checkpoint_path = Path(checkpoint_path).resolve()

        if not self.checkpoint_path.exists() or self.checkpoint_path.stat().st_size < 1024 * 1024 * 10:
            raise FileNotFoundError(
                f"\n[ERROR] SAM checkpoint not found at: {self.checkpoint_path}\n"
                f"Please download the official checkpoint using the provided downloader:\n"
                f"    python scripts/download_sam_checkpoint.py --model-type {self.model_type}\n"
                f"Or place the model weights file at '{self.checkpoint_path}'.\n"
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.model_type not in sam_model_registry:
            raise ValueError(
                f"Unknown SAM model type '{self.model_type}'. Available: {list(sam_model_registry.keys())}"
            )

        print(f"[DentalSAM] Loading SAM ({self.model_type}) from {self.checkpoint_path.name} on {self.device}...")
        self.sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        self.sam.to(device=self.device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
        print("[DentalSAM] SAM initialized successfully.")

    def _generate_dental_prompts(
        self, image_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates deterministic, view-adaptive dental prompts:
        1. Multi-point central dental arch foreground prompts.
        2. Four-corner oral background negative prompts.
        3. Adaptive bounding box enclosing the dental field of view.
        """
        h, w = image_np.shape[:2]

        # In dental photography, teeth are centered along the arch.
        # Foreground positive prompt points (label = 1)
        fg_points = [
            [int(w * 0.50), int(h * 0.50)],  # Center
            [int(w * 0.35), int(h * 0.50)],  # Left arch
            [int(w * 0.65), int(h * 0.50)],  # Right arch
            [int(w * 0.50), int(h * 0.38)],  # Upper occlusal / incisal
            [int(w * 0.50), int(h * 0.62)],  # Lower occlusal / cervical
        ]

        # Background negative prompt points (label = 0) at peripheral image corners
        bg_points = [
            [int(w * 0.05), int(h * 0.05)],  # Top-left corner
            [int(w * 0.95), int(h * 0.05)],  # Top-right corner
            [int(w * 0.05), int(h * 0.95)],  # Bottom-left corner
            [int(w * 0.95), int(h * 0.95)],  # Bottom-right corner
        ]

        points = np.array(fg_points + bg_points, dtype=np.float32)
        labels = np.array([1] * len(fg_points) + [0] * len(bg_points), dtype=np.int32)

        # Central bounding box prior [x_min, y_min, x_max, y_max]
        box = np.array(
            [int(w * 0.10), int(h * 0.10), int(w * 0.90), int(h * 0.90)],
            dtype=np.float32,
        )

        return points, labels, box

    def _postprocess_dental_mask(
        self,
        candidate_masks: np.ndarray,
        iou_scores: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, float]:
        """
        Deterministic dental post-processing:
        1. Select candidate mask based on dental area fraction & IoU score.
        2. Connected component analysis to isolate the main dental crown/arch.
        3. Binary hole filling to close internal specular dental reflections.
        """
        h, w = image_shape
        total_pixels = h * w

        best_mask = None
        best_score = -1.0

        # Sort candidate masks by predicted IoU descending
        sorted_indices = np.argsort(iou_scores)[::-1]

        for idx in sorted_indices:
            mask = candidate_masks[idx].astype(bool)
            area_frac = float(mask.sum()) / total_pixels

            # Filter masks within realistic dental arch coverage range
            if self.min_area_fraction <= area_frac <= self.max_area_fraction:
                best_mask = mask
                best_score = float(iou_scores[idx])
                break

        # Fallback to the top candidate if no mask met the area threshold
        if best_mask is None:
            best_mask = candidate_masks[sorted_indices[0]].astype(bool)
            best_score = float(iou_scores[sorted_indices[0]])

        # Morphological post-processing: Connected Components
        # Keep large connected components (dental crown / arch) and remove small saliva artifacts
        labeled_array, num_features = ndi.label(best_mask)
        if num_features > 1:
            component_sizes = ndi.sum(best_mask, labeled_array, range(1, num_features + 1))
            min_component_size = int(total_pixels * 0.01)  # At least 1% of image
            valid_labels = [
                i + 1 for i, size in enumerate(component_sizes) if size >= min_component_size
            ]
            if valid_labels:
                cleaned_mask = np.isin(labeled_array, valid_labels)
            else:
                # Retain the single largest component
                largest_label = int(np.argmax(component_sizes) + 1)
                cleaned_mask = labeled_array == largest_label
        else:
            cleaned_mask = best_mask

        # Fill internal holes (e.g. enamel flash/specular reflection voids)
        filled_mask = ndi.binary_fill_holes(cleaned_mask)

        # Convert to strict binary uint8 (0=background, 255=foreground)
        final_uint8_mask = (filled_mask.astype(np.uint8)) * 255
        return final_uint8_mask, best_score

    def segment_image(
        self, image_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Segments a single intraoral dental image using SAM.

        Args:
            image_np: RGB image array of shape (H, W, 3), dtype uint8.

        Returns:
            Tuple of:
              - segmented_image: np.ndarray (H, W, 3), background blacked out.
              - mask: np.ndarray (H, W), binary uint8 (0=bg, 255=fg).
              - confidence_score: float predicted IoU from SAM.
        """
        if image_np.ndim == 2:
            image_rgb = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 4:
            image_rgb = image_np[:, :, :3]
        else:
            image_rgb = image_np

        h, w = image_rgb.shape[:2]
        self.predictor.set_image(image_rgb)

        point_coords, point_labels, box = self._generate_dental_prompts(image_rgb)

        candidate_masks, iou_predictions, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        binary_mask, score = self._postprocess_dental_mask(
            candidate_masks, iou_predictions, (h, w)
        )

        # Downstream masked image format
        segmented_image = image_rgb * (binary_mask[:, :, None] // 255)

        return segmented_image, binary_mask, score

    def create_visualization(
        self,
        image_np: np.ndarray,
        mask_uint8: np.ndarray,
        view_name: str = "",
        score: Optional[float] = None,
    ) -> Image.Image:
        """
        Generates a visual validation artifact:
        Original RGB + Transparent Cyan Dental Mask Overlay + White Outline Contour.
        """
        h, w = image_np.shape[:2]
        base_rgb = image_np[:, :, :3] if image_np.ndim == 3 else np.stack([image_np] * 3, axis=-1)
        base_pil = Image.fromarray(base_rgb.astype(np.uint8)).convert("RGBA")

        # Color overlay (Cyan/Teal tint for dental crown)
        overlay_color = np.zeros((h, w, 4), dtype=np.uint8)
        is_fg = mask_uint8 > 128
        overlay_color[is_fg] = [0, 220, 200, 110]  # Semi-transparent cyan
        overlay_pil = Image.fromarray(overlay_color, mode="RGBA")

        # Composite base and overlay
        blended = Image.alpha_composite(base_pil, overlay_pil).convert("RGB")
        draw = ImageDraw.Draw(blended)

        # Draw contour boundary
        struct = ndi.generate_binary_structure(2, 1)
        dilated = ndi.binary_dilation(is_fg, structure=struct)
        boundary = dilated ^ is_fg
        by, bx = np.where(boundary)
        for y, x in zip(by, bx):
            draw.point((x, y), fill=(255, 255, 255))

        # Annotate header
        area_pct = (is_fg.sum() / (h * w)) * 100
        score_text = f" | SAM IoU: {score:.2f}" if score is not None else ""
        header_text = f"View: {view_name} | Crown Area: {area_pct:.1f}%{score_text}"

        # Draw banner at top
        draw.rectangle([(0, 0), (w, 24)], fill=(15, 23, 42))
        draw.text((8, 5), header_text, fill=(241, 245, 249))

        return blended


def segment_dental_sample(
    sample_dir: str | Path,
    segmentor: Optional[DentalSAMSegmentor] = None,
    checkpoint_path: Optional[str | Path] = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes Phase 1 SAM-based segmentation on a complete 5-view sample directory.

    Reads:
      sample_dir / "images" / f"{view}.png"
    Saves:
      sample_dir / "masks" / f"{view}.png" (exact 8-bit binary uint8 PNG)
      sample_dir / "reports" / "sam_segmentation_visualizations" / f"{view}_sam_overlay.png"
      sample_dir / "reports" / "sam_segmentation_contact_sheet.png"

    Returns:
      A dictionary summarizing processed views and metrics.
    """
    sample_path = Path(sample_dir).resolve()
    images_dir = sample_path / "images"
    masks_dir = sample_path / "masks"
    reports_dir = sample_path / "reports"
    vis_dir = reports_dir / "sam_segmentation_visualizations"

    if not images_dir.exists():
        raise FileNotFoundError(f"Input images directory not found at: {images_dir}")

    masks_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if segmentor is None:
        segmentor = DentalSAMSegmentor(
            checkpoint_path=checkpoint_path, model_type=model_type, device=device
        )

    # Search for views
    image_files = sorted(list(images_dir.glob("*.png")))
    if not image_files:
        raise FileNotFoundError(f"No .png images found in: {images_dir}")

    results = {}
    vis_images = []
    view_names = []

    print(f"\n[Phase 1] Running SAM segmentation on {len(image_files)} views in '{sample_path.name}'...")

    for img_path in image_files:
        view_name = img_path.stem
        raw_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(raw_pil, dtype=np.uint8)
        orig_h, orig_w = image_np.shape[:2]

        # Run SAM segmentation
        segmented_img, mask_uint8, score = segmentor.segment_image(image_np)

        # Verify output invariants
        assert mask_uint8.shape == (orig_h, orig_w), f"Mask shape mismatch: {mask_uint8.shape} vs {(orig_h, orig_w)}"
        unique_vals = set(np.unique(mask_uint8))
        assert unique_vals.issubset({0, 255}), f"Mask contains non-binary values: {unique_vals}"

        # Save binary mask in exact location expected by downstream pipeline
        mask_out_path = masks_dir / f"{view_name}.png"
        mask_pil = Image.fromarray(mask_uint8, mode="L")
        mask_pil.save(mask_out_path)

        # Generate visual validation overlay
        vis_pil = segmentor.create_visualization(
            image_np, mask_uint8, view_name=view_name, score=score
        )
        vis_out_path = vis_dir / f"{view_name}_sam_overlay.png"
        vis_pil.save(vis_out_path)

        vis_images.append(vis_pil)
        view_names.append(view_name)

        area_pct = float((mask_uint8 == 255).sum() / (orig_h * orig_w) * 100)
        results[view_name] = {
            "mask_path": str(mask_out_path),
            "vis_path": str(vis_out_path),
            "area_percent": area_pct,
            "iou_confidence": float(score),
            "resolution": [orig_w, orig_h],
        }
        print(f"  [OK] [{view_name}] Crown Area: {area_pct:.1f}% | IoU: {score:.3f} -> {mask_out_path.name}")

    # Build 5-view contact sheet
    if vis_images:
        n_tiles = len(vis_images)
        tile_w, tile_h = 320, 320
        cols = 3
        rows = (n_tiles + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(30, 41, 59))

        for idx, vis_img in enumerate(vis_images):
            r = idx // cols
            c = idx % cols
            thumb = vis_img.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
            sheet.paste(thumb, (c * tile_w, r * tile_h))

        contact_sheet_path = reports_dir / "sam_segmentation_contact_sheet.png"
        sheet.save(contact_sheet_path)
        print(f"[Phase 1] Contact sheet generated -> {contact_sheet_path.name}")

    return {
        "sample_id": sample_path.name,
        "views_processed": len(results),
        "results": results,
    }
