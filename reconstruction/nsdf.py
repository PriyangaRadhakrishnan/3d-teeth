from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree
from skimage import measure


class PositionalEncoding(nn.Module):
    """
    Fourier frequency positional encoding:
    gamma(p) = [p, sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^{L-1} pi p), cos(2^{L-1} pi p)]
    """
    def __init__(self, num_frequencies: int = 6, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.out_dim = 3 * (2 * num_frequencies + (1 if include_input else 0))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 3) 3D coordinate tensor
        Returns:
            encoded: (N, out_dim) frequency-encoded representation
        """
        encodings = [coords] if self.include_input else []
        for i in range(self.num_frequencies):
            freq = (2.0 ** i) * math.pi
            encodings.append(torch.sin(coords * freq))
            encodings.append(torch.cos(coords * freq))
        return torch.cat(encodings, dim=-1)


class NSDFNetwork(nn.Module):
    """
    Neural Signed Distance Field (NSDF) Coordinate Network.

    Learns a continuous scalar field f(x, y, z) -> signed distance (in millimeters/units).
    SIGN CONVENTION (Explicitly Documented):
      - Negative (SDF < 0): Strictly inside the reconstructed dental crown/surface.
      - Zero     (SDF = 0): On the dental surface boundary.
      - Positive (SDF > 0): Strictly outside in free space / oral cavity.
    """
    def __init__(
        self,
        num_frequencies: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4,
        skip_layer: int = 2
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(num_frequencies=num_frequencies, include_input=True)
        in_dim = self.pos_encoder.out_dim
        self.hidden_dim = hidden_dim
        self.skip_layer = skip_layer

        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim + in_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, 1)

        # Softplus with beta=100 provides smooth, non-zero second derivatives for Eikonal loss
        self.activation = nn.Softplus(beta=100)

        # Geometric initialization
        self._init_weights()

    def _init_weights(self):
        for m in [self.layer1, self.layer2, self.layer3, self.layer4]:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        nn.init.normal_(self.out_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.out_layer.bias, 0.0)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the scalar signed distance at arbitrary 3D positions.
        Args:
            coords: (N, 3) 3D coordinate points
        Returns:
            sdf: (N, 1) signed distance values
        """
        enc = self.pos_encoder(coords)
        h = self.activation(self.layer1(enc))
        h = self.activation(self.layer2(h))
        # Skip connection
        h = self.activation(self.layer3(torch.cat([h, enc], dim=-1)))
        h = self.activation(self.layer4(h))
        sdf = self.out_layer(h)
        return sdf

    def compute_gradient(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Analytically computes the spatial gradient of the SDF field:
        nabla_x f(x) = [df/dx, df/dy, df/dz]
        """
        was_tracking = coords.requires_grad
        coords = coords.requires_grad_(True)
        sdf = self.forward(coords)
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=coords,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        if not was_tracking:
            coords.requires_grad_(False)
        return grad


class NeuralSignedDistanceField:
    """
    High-level NSDF Manager handling training sample generation, network training,
    SDF query, surface extraction via Marching Cubes, and model persistence.
    """
    SIGN_CONVENTION = "Negative = Inside Dental Surface, Zero = Dental Surface, Positive = Outside Free Space"

    def __init__(
        self,
        hidden_dim: int = 128,
        num_frequencies: int = 6,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NSDFNetwork(
            num_frequencies=num_frequencies,
            hidden_dim=hidden_dim
        ).to(self.device)
        self.bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def generate_training_data(
        self,
        surface_points: np.ndarray,
        surface_normals: Optional[np.ndarray] = None,
        camera_positions: Optional[List[np.ndarray]] = None,
        num_surface_samples: int = 8000,
        num_near_samples: int = 8000,
        num_far_samples: int = 4000
    ) -> Dict[str, torch.Tensor]:
        """
        Generates genuine 3D supervision samples from reconstructed dental geometry:
        1. Exact surface points (P_0) with SDF = 0 and normal supervision.
        2. Near-surface perturbed points (P_near) displaced along normals:
           - positive offset -> outside (SDF = +delta)
           - negative offset -> inside (SDF = -delta)
        3. Far free-space points (P_far) sampled uniformly across the bounding box,
           with distance to nearest surface and sign determined by camera sightlines.
        """
        N = len(surface_points)
        assert N > 10, f"Insufficient surface points: {N}"

        # Subsample surface points
        if N > num_surface_samples:
            idx = np.random.choice(N, num_surface_samples, replace=False)
            pts_surf = surface_points[idx].astype(np.float32)
        else:
            pts_surf = surface_points.astype(np.float32)

        # Estimate / orient normals if not provided
        if surface_normals is None or len(surface_normals) != len(surface_points):
            pcd = trimesh.PointCloud(pts_surf)
            # KDTree-based normal estimation
            tree = cKDTree(pts_surf)
            _, neighbors = tree.query(pts_surf, k=15)
            normals = np.zeros_like(pts_surf)
            for i in range(len(pts_surf)):
                cov = np.cov(pts_surf[neighbors[i]].T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                normals[i] = eigvecs[:, 0]
        else:
            if N > num_surface_samples:
                normals = surface_normals[idx].astype(np.float32)
            else:
                normals = surface_normals.astype(np.float32)

        # Orient normals outward toward camera positions
        if camera_positions and len(camera_positions) > 0:
            cam_centers = np.array(camera_positions)
            # Find nearest camera for each point
            for i in range(len(pts_surf)):
                cam_dists = np.linalg.norm(cam_centers - pts_surf[i], axis=-1)
                best_cam = cam_centers[np.argmin(cam_dists)]
                ray_to_cam = best_cam - pts_surf[i]
                if np.dot(normals[i], ray_to_cam) < 0:
                    normals[i] = -normals[i]

        norm_magnitudes = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.where(norm_magnitudes > 1e-6, normals / np.maximum(norm_magnitudes, 1e-6), np.array([0, 0, 1], dtype=np.float32))

        # 1. Surface points: SDF = 0.0
        p0 = pts_surf
        d0 = np.zeros((len(p0), 1), dtype=np.float32)
        n0 = normals

        # 2. Near-surface samples: small displacements along surface normals
        # Sign convention: +offset is outside (along normal), -offset is inside (opposite normal)
        deltas = np.random.normal(loc=0.0, scale=0.02, size=(num_near_samples, 1)).astype(np.float32)
        # Add small isotropic perturbation
        sub_idx = np.random.choice(len(pts_surf), num_near_samples, replace=True)
        jitter = np.random.normal(scale=0.005, size=(num_near_samples, 3)).astype(np.float32)
        p_near = pts_surf[sub_idx] + deltas * normals[sub_idx] + jitter
        d_near = deltas  # positive = outside, negative = inside

        # 3. Far free-space points: uniform samples in padded bounding box
        bbox_min = pts_surf.min(axis=0) - 0.15 * (pts_surf.max(axis=0) - pts_surf.min(axis=0))
        bbox_max = pts_surf.max(axis=0) + 0.15 * (pts_surf.max(axis=0) - pts_surf.min(axis=0))
        p_far = np.random.uniform(bbox_min, bbox_max, size=(num_far_samples, 3)).astype(np.float32)

        # Unsigned distance via KDTree
        kdtree = cKDTree(pts_surf)
        dists, nearest_idx = kdtree.query(p_far)
        dists = dists[:, None].astype(np.float32)

        # Determine sign for far points via ray projection from nearest surface point normal
        diff = p_far - pts_surf[nearest_idx]
        dot_prod = (diff * normals[nearest_idx]).sum(axis=-1, keepdims=True)
        signs = np.where(dot_prod >= 0, 1.0, -1.0).astype(np.float32)
        d_far = signs * dists

        # Combine training tensors
        all_coords = np.vstack([p0, p_near, p_far]).astype(np.float32)
        all_sdf = np.vstack([d0, d_near, d_far]).astype(np.float32)

        return {
            "coords": torch.from_numpy(all_coords).to(self.device),
            "sdf_gt": torch.from_numpy(all_sdf).to(self.device),
            "surface_coords": torch.from_numpy(p0).to(self.device),
            "surface_normals": torch.from_numpy(n0).to(self.device),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "counts": {
                "surface_points": len(p0),
                "near_points": len(p_near),
                "far_points": len(p_far),
                "total_samples": len(all_coords)
            }
        }

    def train_nsdf(
        self,
        training_data: Dict[str, Any],
        iterations: int = 150,
        lr: float = 1e-3,
        batch_size: int = 2048,
        lambda_zero: float = 5.0,
        lambda_dist: float = 2.0,
        lambda_eikonal: float = 0.1,
        lambda_normal: float = 0.5
    ) -> Dict[str, Any]:
        """
        Fits the Neural Signed Distance Field using composite SDF objectives:
        - Surface zero-level loss
        - Space distance regression loss
        - Eikonal differential regularization (||grad f|| = 1)
        - Normal direction alignment on surface points
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-5)

        coords_all = training_data["coords"]
        sdf_gt_all = training_data["sdf_gt"]
        surf_coords_all = training_data["surface_coords"]
        surf_norm_all = training_data["surface_normals"]
        total_samples = len(coords_all)
        total_surf = len(surf_coords_all)

        self.bounds = (training_data["bbox_min"], training_data["bbox_max"])

        history = {"loss": [], "sdf_loss": [], "eikonal_loss": [], "normal_loss": []}
        start_time = time.time()

        for step in range(iterations):
            optimizer.zero_grad(set_to_none=True)

            # Mini-batch sampling
            idx = torch.randint(0, total_samples, (min(batch_size, total_samples),), device=self.device)
            b_coords = coords_all[idx]
            b_sdf_gt = sdf_gt_all[idx]

            s_idx = torch.randint(0, total_surf, (min(batch_size // 2, total_surf),), device=self.device)
            b_surf = surf_coords_all[s_idx]
            b_norm = surf_norm_all[s_idx]

            # 1. SDF Distance Loss
            pred_sdf = self.model(b_coords)
            loss_dist = F.smooth_l1_loss(pred_sdf, b_sdf_gt)

            # 2. Surface Zero-level Loss
            pred_zero = self.model(b_surf)
            loss_zero = F.l1_loss(pred_zero, torch.zeros_like(pred_zero))

            # 3. Eikonal Regularization: ||grad f|| ≈ 1
            grad = self.model.compute_gradient(b_coords)
            grad_norm = torch.norm(grad, dim=-1, keepdim=True)
            loss_eikonal = F.mse_loss(grad_norm, torch.ones_like(grad_norm))

            # 4. Normal Alignment Loss on surface
            surf_grad = self.model.compute_gradient(b_surf)
            surf_grad_norm = surf_grad / torch.clamp(torch.norm(surf_grad, dim=-1, keepdim=True), min=1e-6)
            loss_norm = (1.0 - (surf_grad_norm * b_norm).sum(dim=-1)).mean()

            # Total Loss
            total_loss = (
                lambda_zero * loss_zero +
                lambda_dist * loss_dist +
                lambda_eikonal * loss_eikonal +
                lambda_normal * loss_norm
            )

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Track finite losses
            l_val = float(total_loss.item())
            history["loss"].append(l_val)
            history["sdf_loss"].append(float(loss_dist.item() + loss_zero.item()))
            history["eikonal_loss"].append(float(loss_eikonal.item()))
            history["normal_loss"].append(float(loss_norm.item()))

            if (step + 1) % 50 == 0 or (step + 1) == iterations:
                print(f"[NSDF] Iteration {step+1}/{iterations} | Loss: {l_val:.5f} | Eikonal: {loss_eikonal.item():.5f} | Norm: {loss_norm.item():.5f}")

        elapsed = time.time() - start_time
        self.model.eval()

        return {
            "initial_loss": history["loss"][0],
            "final_loss": history["loss"][-1],
            "final_eikonal_loss": history["eikonal_loss"][-1],
            "final_normal_loss": history["normal_loss"][-1],
            "iterations": iterations,
            "training_time_sec": elapsed,
            "sample_counts": training_data["counts"],
            "sign_convention": self.SIGN_CONVENTION
        }

    @torch.no_grad()
    def query_sdf(self, query_points: Union[np.ndarray, torch.Tensor], batch_size: int = 8192) -> np.ndarray:
        """
        Queries the learned signed distance field at arbitrary 3D spatial points.
        """
        self.model.eval()
        if isinstance(query_points, torch.Tensor):
            pts = query_points.detach().cpu().numpy()
        else:
            pts = np.asarray(query_points)
        N = len(pts)
        results = []
        for i in range(0, N, batch_size):
            chunk = torch.from_numpy(pts[i:i + batch_size].astype(np.float32)).to(self.device)
            sdf_val = self.model(chunk).detach().cpu().numpy()
            results.append(sdf_val)
        return np.vstack(results).flatten()

    def extract_surface(
        self,
        resolution: int = 64,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        level: float = 0.0,
        output_ply: Optional[Union[str, Path]] = None,
        output_obj: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Extracts the zero-level iso-surface (SDF = 0.0) from the continuous neural field
        using Marching Cubes, exporting clean triangular mesh models.
        """
        if bounds is None:
            if self.bounds is not None:
                bounds = self.bounds
            else:
                raise ValueError("Bounding box bounds must be specified for surface extraction")

        bbox_min, bbox_max = bounds
        xs = np.linspace(bbox_min[0], bbox_max[0], resolution, dtype=np.float32)
        ys = np.linspace(bbox_min[1], bbox_max[1], resolution, dtype=np.float32)
        zs = np.linspace(bbox_min[2], bbox_max[2], resolution, dtype=np.float32)

        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_pts = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=-1)

        # Dense evaluation
        sdf_vals = self.query_sdf(grid_pts, batch_size=16384)
        volume = sdf_vals.reshape(resolution, resolution, resolution)

        dx = (bbox_max[0] - bbox_min[0]) / max(resolution - 1, 1)
        dy = (bbox_max[1] - bbox_min[1]) / max(resolution - 1, 1)
        dz = (bbox_max[2] - bbox_min[2]) / max(resolution - 1, 1)
        spacing = (dx, dy, dz)

        # Marching cubes extraction at zero-level boundary
        verts, faces, normals, _ = measure.marching_cubes(
            volume,
            level=level,
            spacing=spacing
        )

        # Shift vertices to global world coordinates
        verts = verts + bbox_min

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

        # Export mesh files
        if output_ply:
            out_ply = Path(output_ply).resolve()
            out_ply.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(out_ply))

        if output_obj:
            out_obj = Path(output_obj).resolve()
            out_obj.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(out_obj))

        return {
            "vertices_count": int(len(mesh.vertices)),
            "faces_count": int(len(mesh.faces)),
            "is_watertight": bool(mesh.is_watertight),
            "bounds": [mesh.bounds[0].tolist(), mesh.bounds[1].tolist()],
            "extent": (mesh.bounds[1] - mesh.bounds[0]).tolist(),
            "output_ply": str(output_ply) if output_ply else None,
            "output_obj": str(output_obj) if output_obj else None
        }

    def save_checkpoint(self, path: Union[str, Path]):
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "bounds": self.bounds,
            "sign_convention": self.SIGN_CONVENTION
        }, p)

    def load_checkpoint(self, path: Union[str, Path]):
        p = Path(path).resolve()
        checkpoint = torch.load(p, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.bounds = checkpoint.get("bounds")


def train_and_extract_nsdf(
    sample_dir: Union[str, Path],
    device: Optional[str] = None,
    iterations: int = 150,
    resolution: int = 64
) -> Dict[str, Any]:
    """
    High-level entry point called directly by the pipeline runner immediately following
    Multiscale Gaussian Rendering.
    """
    sample_path = Path(sample_dir).resolve()
    ply_path = sample_path / "reconstruction.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"reconstruction.ply not found at: {ply_path}")

    # 1. Load reconstructed geometry as supervision
    cloud = trimesh.load(ply_path, process=False)
    pts = np.asarray(cloud.vertices, dtype=np.float32)

    # 2. Load camera positions to orient normals
    metadata_file = sample_path / "metadata.json"
    cam_positions = []
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        cam_file = sample_path / metadata.get("camera_file", "cameras/cameras.json")
        if cam_file.exists():
            cams = json.loads(cam_file.read_text(encoding="utf-8"))
            cam_positions = [np.array(cams[v]["position"], dtype=np.float32) for v in cams if "position" in cams[v]]

    print(f"[NSDF] Initializing Neural Signed Distance Field for {sample_path.name}...")
    nsdf = NeuralSignedDistanceField(device=device)

    # 3. Generate supervision samples
    print(f"[NSDF] Generating supervision point distributions (surface, near-surface, free-space)...")
    training_data = nsdf.generate_training_data(
        surface_points=pts,
        camera_positions=cam_positions,
        num_surface_samples=6000,
        num_near_samples=6000,
        num_far_samples=3000
    )

    # 4. Train the continuous field
    print(f"[NSDF] Fitting continuous signed-distance field ({iterations} iterations)...")
    train_metrics = nsdf.train_nsdf(training_data, iterations=iterations)

    # 5. Save model checkpoint
    model_path = sample_path / "nsdf_model.pth"
    nsdf.save_checkpoint(model_path)
    print(f"[NSDF] Saved model weights to {model_path.name}")

    # 6. Extract zero-level isosurface (SDF = 0.0) via Marching Cubes
    print(f"[NSDF] Extracting zero-level boundary surface (Marching Cubes resolution={resolution}^3)...")
    surf_ply = sample_path / "nsdf_surface.ply"
    surf_obj = sample_path / "nsdf_surface.obj"
    mesh_metrics = nsdf.extract_surface(
        resolution=resolution,
        level=0.0,
        output_ply=surf_ply,
        output_obj=surf_obj
    )
    print(f"[NSDF] Extracted surface mesh: {mesh_metrics['vertices_count']} vertices, {mesh_metrics['faces_count']} faces -> {surf_obj.name}")

    # 7. Record NSDF comprehensive metrics artifact
    full_metrics = {
        "sample_id": sample_path.name,
        "sign_convention": nsdf.SIGN_CONVENTION,
        "model_file": str(model_path.name),
        "surface_ply": str(surf_ply.name),
        "surface_obj": str(surf_obj.name),
        "training": train_metrics,
        "surface_mesh": mesh_metrics
    }
    metrics_path = sample_path / "nsdf_metrics.json"
    metrics_path.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    print(f"[NSDF] Saved metrics report to {metrics_path.name}")

    return full_metrics
