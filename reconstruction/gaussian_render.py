"""
Phase 6: Multiscale Gaussian Rendering Module.

Implements multi-scale Gaussian rendering supporting multiple resolutions:
- Scale 1.0x (640x640)
- Scale 0.5x (320x320)
- Scale 0.25x (160x160)

Provides:
- CUDA gsplat rasterization when available
- CPU projection-based Gaussian point splatting fallback when CUDA is unavailable
- Comprehensive rendering metrics recording:
  * rendering resolution(s)
  * number of Gaussians
  * rendering time per scale and total
  * memory usage
- Saves multiscale rendered outputs in:
  * sample_dir / "multiscale" / ...
  * sample_dir / "gaussian_renders" / ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import psutil
import torch
from PIL import Image, ImageDraw


def project_gaussians_cpu(
    means: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    viewmat: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    bg_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    CPU projection-based Gaussian point splatting renderer.
    Renders 3D Gaussians onto a 2D image plane via projective splatting and alpha compositing.
    """
    N = len(means)
    # Homogeneous coordinates (N, 4)
    homo = np.hstack([means, np.ones((N, 1), dtype=np.float32)])
    # Transform to camera frame: P_cam = homo @ viewmat.T
    cam_pts = homo @ viewmat.T
    cam_xyz = cam_pts[:, :3]
    z = cam_xyz[:, 2]

    # Front-facing mask
    valid = z > 0.05
    if not np.any(valid):
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    valid_indices = np.where(valid)[0]
    # Sort back-to-front (descending depth) for alpha blending
    sort_order = valid_indices[np.argsort(-z[valid_indices])]

    cam_xyz_sorted = cam_xyz[sort_order]
    z_sorted = z[sort_order]
    colors_sorted = colors[sort_order]
    opacities_sorted = opacities[sort_order]
    scales_sorted = scales[sort_order]

    # Perspective projection
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = (cam_xyz_sorted[:, 0] * fx / z_sorted + cx).astype(np.int32)
    v = (cam_xyz_sorted[:, 1] * fy / z_sorted + cy).astype(np.int32)

    # Splat radii in pixels
    mean_scale = np.mean(scales_sorted, axis=-1)
    radii = np.clip((mean_scale * max(fx, fy) / np.maximum(z_sorted, 0.05)).astype(np.int32), 1, 8)

    canvas = np.full((height, width, 3), bg_color, dtype=np.float32)

    # Alpha composite splats
    for i in range(len(sort_order)):
        ui, vi, r = u[i], v[i], radii[i]
        if ui + r < 0 or ui - r >= width or vi + r < 0 or vi - r >= height:
            continue

        c = colors_sorted[i]
        alpha = float(np.clip(opacities_sorted[i], 0.05, 0.85))

        x_min = max(0, ui - r)
        x_max = min(width, ui + r + 1)
        y_min = max(0, vi - r)
        y_max = min(height, vi + r + 1)

        # Soft circular falloff
        ys, xs = np.ogrid[y_min - vi:y_max - vi, x_min - ui:x_max - ui]
        mask = (xs**2 + ys**2) <= r**2
        if not np.any(mask):
            continue

        weight = alpha * (1.0 - np.sqrt(xs**2 + ys**2) / (r + 1e-4))
        weight = np.clip(weight, 0.0, 1.0)[:, :, None] * mask[:, :, None]

        canvas[y_min:y_max, x_min:x_max] = (
            canvas[y_min:y_max, x_min:x_max] * (1.0 - weight) + c[None, None, :] * weight
        )

    return (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_model_multiscale(
    sample_dir: Union[str, Path],
    resolutions: Optional[List[Tuple[int, int]]] = None,
    device: Optional[str] = None,
    max_points: Optional[int] = None
) -> Dict[str, Any]:
    """
    Renders the trained 3D Gaussian model across multiple resolutions.
    
    Default resolutions:
    - (640, 640): Full scale 1.0x
    - (320, 320): Half scale 0.5x
    - (160, 160): Quarter scale 0.25x
    """
    sample = Path(sample_dir).resolve()
    target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_path = sample / "gaussian_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Gaussian model checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    cameras = json.loads((sample / metadata["camera_file"]).read_text(encoding="utf-8"))

    means = checkpoint["means"].to(target_device)
    colors = checkpoint["colors"].to(target_device)
    scales = checkpoint["scales"].to(target_device)
    opacities = checkpoint["opacities"].to(target_device)
    quats = checkpoint["quats"].to(target_device)
    view_names = metadata["views"]

    if max_points is not None and len(means) > max_points:
        sub_idx = torch.linspace(0, len(means) - 1, max_points, dtype=torch.long)
        means = means[sub_idx]
        colors = colors[sub_idx]
        scales = scales[sub_idx]
        opacities = opacities[sub_idx]
        quats = quats[sub_idx]

    num_gaussians = int(len(means))

    blender_viewmats = torch.tensor([cameras[v]["extrinsics_world_to_camera"] for v in view_names], dtype=torch.float32, device=target_device)
    axis_flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=target_device))
    viewmats = axis_flip.unsqueeze(0) @ blender_viewmats
    Ks_orig = torch.tensor([cameras[v]["intrinsics"] for v in view_names], dtype=torch.float32, device=target_device)

    orig_w, orig_h = metadata.get("resolution", [640, 640])
    if resolutions is None:
        resolutions = [
            (orig_w, orig_h),
            (orig_w // 2, orig_h // 2),
            (orig_w // 4, orig_h // 4)
        ]

    has_cuda_gsplat = False
    try:
        from gsplat.rendering import rasterization
        if str(target_device) == "cuda":
            from gsplat.cuda import _backend
            if _backend._C is not None:
                has_cuda_gsplat = True
    except Exception:
        has_cuda_gsplat = False

    multiscale_dir = sample / "multiscale"
    multiscale_dir.mkdir(parents=True, exist_ok=True)
    primary_renders_dir = sample / "gaussian_renders"
    primary_renders_dir.mkdir(parents=True, exist_ok=True)

    multiscale_results = {}
    rendering_timings = {}
    t_start_total = time.perf_counter()

    means_np = means.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()
    scales_np = scales.detach().cpu().numpy()
    opacities_np = opacities.detach().cpu().numpy()
    viewmats_np = viewmats.detach().cpu().numpy()
    Ks_np = Ks_orig.detach().cpu().numpy()

    for width, height in resolutions:
        scale_label = f"{width}x{height}"
        scale_out_dir = multiscale_dir / f"scale_{scale_label}"
        scale_out_dir.mkdir(parents=True, exist_ok=True)

        sx = width / float(orig_w)
        sy = height / float(orig_h)

        t0 = time.perf_counter()
        scale_images = {}

        if has_cuda_gsplat:
            # Scaled intrinsics
            Ks_scaled = Ks_orig.clone()
            Ks_scaled[:, 0, :] *= sx
            Ks_scaled[:, 1, :] *= sy
            from gsplat.rendering import rasterization
            rendered, _, _ = rasterization(
                means, quats, scales, opacities, colors, viewmats, Ks_scaled, width, height,
                backgrounds=torch.zeros(3, device=target_device), camera_model="pinhole"
            )
            for idx, v in enumerate(view_names):
                arr = (rendered[idx].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                scale_images[v] = arr
        else:
            # Software point splatting
            for idx, v in enumerate(view_names):
                K_scaled = Ks_np[idx].copy()
                K_scaled[0, :] *= sx
                K_scaled[1, :] *= sy
                arr = project_gaussians_cpu(
                    means=means_np,
                    colors=colors_np,
                    scales=scales_np,
                    opacities=opacities_np,
                    viewmat=viewmats_np[idx],
                    K=K_scaled,
                    width=width,
                    height=height
                )
                scale_images[v] = arr

        render_sec = time.perf_counter() - t0
        rendering_timings[scale_label] = round(render_sec, 4)

        # Save per-view rendered images
        tiles = []
        for v, img_arr in scale_images.items():
            img_pil = Image.fromarray(img_arr)
            img_pil.save(scale_out_dir / f"{v}.png")
            # If primary full-scale resolution, also save to gaussian_renders/ for downstream compatibility
            if (width, height) == (orig_w, orig_h):
                img_pil.save(primary_renders_dir / f"{v}.png")
            tiles.append(img_pil.resize((320, 320)))

        # Contact sheet for this scale
        sheet = Image.new("RGB", (960, 700), "#18181b")
        for idx, tile in enumerate(tiles):
            sheet.paste(tile, ((idx % 3) * 320, (idx // 3) * 350))
        sheet.save(scale_out_dir / "contact_sheet.png")
        if (width, height) == (orig_w, orig_h):
            sheet.save(primary_renders_dir / "contact_sheet.png")

        multiscale_results[scale_label] = {
            "resolution": [width, height],
            "rendering_time_sec": round(render_sec, 4),
            "views_rendered": list(scale_images.keys()),
            "output_directory": str(scale_out_dir)
        }

    total_render_time = time.perf_counter() - t_start_total

    # Record memory usage
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    mem_info = {
        "cpu_ram_used_gb": round(proc.memory_info().rss / (1024**3), 3),
        "system_ram_used_gb": round(vm.used / (1024**3), 3),
        "cuda_memory": "CUDA unavailable" if str(target_device) == "cpu" else f"{torch.cuda.max_memory_allocated() / (1024**3):.3f} GB"
    }

    full_metrics = {
        "sample_id": sample.name,
        "device": str(target_device),
        "backend": "CUDA gsplat" if has_cuda_gsplat else "CPU Projective Splatting",
        "num_gaussians": num_gaussians,
        "rendering_resolutions": [[w, h] for w, h in resolutions],
        "total_rendering_time_sec": round(total_render_time, 4),
        "rendering_timings_per_scale": rendering_timings,
        "memory_usage": mem_info,
        "scales": multiscale_results
    }

    metrics_file = sample / "multiscale_rendering_metrics.json"
    metrics_file.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    print(f"[Multiscale Rendering] Rendered {len(resolutions)} scales ({num_gaussians} Gaussians) in {total_render_time:.2f}s -> {metrics_file.name}")

    return full_metrics


def render_model(sample_dir: str | Path, device: str = None) -> Path:
    """Backward-compatible entry point."""
    res = render_model_multiscale(sample_dir, device=device)
    return Path(sample_dir) / "gaussian_renders"


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiscale Gaussian Rendering")
    parser.add_argument("sample")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(json.dumps(render_model_multiscale(args.sample, device=args.device), indent=2))


if __name__ == "__main__":
    main()