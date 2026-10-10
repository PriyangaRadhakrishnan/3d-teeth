from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from trimesh import load


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cloud = load(path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float32)
    colors = np.asarray(cloud.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
    return points, colors


class NeuralRepresentation(torch.nn.Module):
    """
    Phase 4: Learnable Neural Representation integrating Fused Cross-View Features.
    
    Receives:
    1. 3D coordinates (x, y, z) of Gaussian primitives
    2. Multi-view fused feature vectors sampled from the Cross-View Feature Fusion stage.
    
    Predicts:
    Decoupled view-independent diffuse color and radiance for each 3D Gaussian primitive.
    """
    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(3 + feat_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 3)
        )

    def forward(self, positions: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features is not None and features.shape[-1] == self.feat_dim:
            inp = torch.cat([positions, features], dim=-1)
        else:
            dummy = torch.zeros((positions.shape[0], self.feat_dim), device=positions.device, dtype=positions.dtype)
            inp = torch.cat([positions, dummy], dim=-1)
        return self.network(inp)


def sample_cross_view_features_at_points(
    means: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    fused_features: torch.Tensor,
    target_dim: int = 64
) -> torch.Tensor:
    """
    Projects 3D point coordinates into the calibrated views and samples
    the fused multi-view feature maps at the projected coordinates via bilinear interpolation.
    """
    N = means.shape[0]
    V, C, H, W = fused_features.shape
    device = means.device

    ones = torch.ones((N, 1), device=device, dtype=means.dtype)
    pts_homo = torch.cat([means, ones], dim=-1).unsqueeze(0).expand(V, -1, -1)

    # P_cam = pts_homo @ viewmats.T
    P_cam = torch.bmm(pts_homo, viewmats.transpose(1, 2))[:, :, :3]
    z_cam = P_cam[:, :, 2:3]

    p_pix = torch.bmm(P_cam, Ks.transpose(1, 2))
    safe_z = torch.clamp(z_cam, min=1e-4)
    u = p_pix[:, :, 0:1] / safe_z
    v = p_pix[:, :, 1:2] / safe_z

    grid_u = (u / max(W - 1, 1)) * 2.0 - 1.0
    grid_v = (v / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.cat([grid_u, grid_v], dim=-1).view(V, N, 1, 2)

    sampled = F.grid_sample(fused_features, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.squeeze(-1).permute(2, 0, 1)  # (N, V, C)

    valid = (z_cam.squeeze(-1).transpose(0, 1) > 0.05) & (torch.abs(grid_u.squeeze(-1).transpose(0, 1)) <= 1.0) & (torch.abs(grid_v.squeeze(-1).transpose(0, 1)) <= 1.0)
    valid_f = valid.unsqueeze(-1).float()

    sum_feats = (sampled * valid_f).sum(dim=1)
    denom = torch.clamp(valid_f.sum(dim=1), min=1.0)
    point_feats = sum_feats / denom

    if point_feats.shape[-1] >= target_dim:
        point_feats = point_feats[:, :target_dim]
    else:
        point_feats = F.pad(point_feats, (0, target_dim - point_feats.shape[-1]))

    return point_feats


def validate_gaussian_parameters(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor
) -> Dict[str, bool]:
    """
    Phase 5 Verification: Validates that all 3D Gaussian parameters are finite,
    non-degenerate, and physically consistent.
    """
    assert torch.isfinite(means).all(), "Non-finite Gaussian positions (means)"
    assert torch.isfinite(scales).all() and (scales > 0).all(), "Invalid Gaussian scales"
    assert torch.isfinite(quats).all(), "Non-finite Gaussian quaternions"
    assert torch.isfinite(opacities).all() and (opacities >= 0).all() and (opacities <= 1.0).all(), "Invalid opacities"
    assert torch.isfinite(colors).all() and (colors >= 0).all() and (colors <= 1.0).all(), "Invalid colors"
    return {
        "positions_valid": True,
        "scales_valid": True,
        "quats_valid": True,
        "opacities_valid": True,
        "colors_valid": True
    }


def train(sample_dir: str | Path, iterations: int = 500, max_points: int = 35000, device: str = "cpu") -> dict:
    sample = Path(sample_dir)
    target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    has_gsplat = False
    try:
        from gsplat.rendering import rasterization
        if str(target_device) == "cuda":
            from gsplat.cuda import _backend
            if _backend._C is not None:
                has_gsplat = True
    except Exception as e:
        print(f"[Gaussian Training] gsplat CUDA rasterizer unavailable ({e}), using CPU neural field optimization.")

    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    cameras = json.loads((sample / metadata["camera_file"]).read_text(encoding="utf-8"))
    points, initial_colors = load_ply(sample / "reconstruction.ply")
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, initial_colors = points[indices], initial_colors[indices]

    means = torch.from_numpy(points).to(target_device)

    view_names = metadata["views"]
    blender_viewmats = torch.tensor([cameras[view]["extrinsics_world_to_camera"] for view in view_names], dtype=torch.float32, device=target_device)
    blender_to_opencv = torch.eye(4, dtype=torch.float32, device=target_device)
    blender_to_opencv[1, 1] = -1.0
    blender_to_opencv[2, 2] = -1.0
    viewmats = blender_to_opencv.unsqueeze(0) @ blender_viewmats
    Ks = torch.tensor([cameras[view]["intrinsics"] for view in view_names], dtype=torch.float32, device=target_device)

    # Phase 4: Load Fused Cross-View Features and integrate into Neural Representation
    fused_features_tensor = None
    has_fused_features = False
    cross_feat_path = sample / "cross_view_features.pt"
    feat_path = sample / "features.pt"

    load_path = cross_feat_path if cross_feat_path.exists() else (feat_path if feat_path.exists() else None)
    if load_path:
        try:
            f_data = torch.load(load_path, map_location=target_device, weights_only=False)
            f_multiscale = f_data.get("fused_multiscale_features") or f_data.get("cross_view_fused_features") or f_data.get("multiscale_features")
            if f_multiscale and "scale_1_8" in f_multiscale:
                fused_features_tensor = f_multiscale["scale_1_8"].to(target_device, dtype=torch.float32)
                has_fused_features = True
                print(f"[Gaussian Training] Successfully integrated Fused Cross-View Features (shape: {list(fused_features_tensor.shape)})")
        except Exception as e:
            print(f"[Gaussian Training] Warning loading fused features: {e}")

    feat_dim = 64
    point_features = None
    if has_fused_features and fused_features_tensor is not None:
        point_features = sample_cross_view_features_at_points(
            means=means,
            viewmats=viewmats,
            Ks=Ks,
            fused_features=fused_features_tensor,
            target_dim=feat_dim
        )

    neural_field = NeuralRepresentation(feat_dim=feat_dim).to(target_device)

    scales = torch.full((len(points), 3), 0.004, device=target_device)
    opacities = torch.full((len(points),), 0.35, device=target_device)
    quats = torch.zeros((len(points), 4), device=target_device)
    quats[:, 0] = 1.0

    optimizer = torch.optim.Adam(neural_field.parameters(), lr=0.005)

    targets = torch.stack([torch.from_numpy(np.asarray(Image.open(sample / "images" / f"{view}.png").convert("RGB"), dtype=np.float32) / 255.0) for view in view_names]).to(target_device)
    masks = torch.stack([torch.from_numpy(np.asarray(Image.open(sample / "masks" / f"{view}.png").convert("L"), dtype=np.float32) / 255.0) for view in view_names]).to(target_device)

    backgrounds = torch.zeros(3, device=target_device)
    height, width = targets.shape[1:3]
    losses = []
    initial_colors_tensor = torch.from_numpy(initial_colors).to(target_device)

    for step in range(iterations):
        predicted_colors = neural_field(means, point_features)

        if has_gsplat:
            from gsplat.rendering import rasterization
            rendered, _, _ = rasterization(
                means, quats, scales, opacities, torch.sigmoid(predicted_colors),
                viewmats, Ks, width, height, backgrounds=backgrounds, camera_model="pinhole"
            )
            pixel_loss = torch.nn.functional.smooth_l1_loss(rendered, targets, reduction="none").mean(dim=-1)
            foreground_weight = 0.2 + 0.8 * masks
            loss = (pixel_loss * foreground_weight).sum() / foreground_weight.sum()
        else:
            color_loss = torch.nn.functional.mse_loss(torch.sigmoid(predicted_colors), initial_colors_tensor)
            loss = color_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 50 == 0 or (step + 1) == iterations:
            print(f"[Gaussian Training] iteration {step + 1}/{iterations} loss={losses[-1]:.6f}")

    final_colors = torch.sigmoid(neural_field(means, point_features)).detach().cpu()

    # Validate Gaussian parameters (Phase 5)
    validate_gaussian_parameters(
        means=means.detach().cpu(),
        scales=scales.cpu(),
        quats=quats.cpu(),
        opacities=opacities.cpu(),
        colors=final_colors
    )

    output = sample / "gaussian_model.pt"
    torch.save({
        "means": means.detach().cpu(),
        "colors": final_colors,
        "scales": scales.cpu(),
        "opacities": opacities.cpu(),
        "quats": quats.cpu(),
        "view_names": view_names,
        "camera_file": metadata["camera_file"],
        "fused_cross_view_features_integrated": has_fused_features,
        "feature_dim": feat_dim
    }, output)

    result = {
        "mode": "fused-cross-view-neural-field-gaussian-optimization",
        "device": str(target_device),
        "iterations": iterations,
        "points": len(points),
        "fused_features_integrated": has_fused_features,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "output": str(output)
    }
    (sample / "gaussian_training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gaussian Neural Fields with Fused Cross-View Features")
    parser.add_argument("sample")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(json.dumps(train(args.sample, args.iterations, args.max_points, args.device), indent=2))


if __name__ == "__main__":
    main()