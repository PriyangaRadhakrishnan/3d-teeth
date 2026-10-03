from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


def compute_mesh_curvatures(
    mesh: trimesh.Trimesh,
    k_neighbors: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates principal curvatures (k1, k2), mean curvature H, and Gaussian curvature K
    at each vertex of a 3D dental surface mesh using local quadric surface fitting.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    N = len(vertices)

    tree = cKDTree(vertices)
    _, neighbor_indices = tree.query(vertices, k=min(k_neighbors, N))

    mean_curvatures = np.zeros(N, dtype=np.float64)
    gaussian_curvatures = np.zeros(N, dtype=np.float64)

    for i in range(N):
        nbrs = vertices[neighbor_indices[i]]
        n = normals[i]

        # Construct local tangent plane coordinate frame (u, v, n)
        if abs(n[0]) > abs(n[1]):
            u = np.array([-n[2], 0.0, n[0]], dtype=np.float64)
        else:
            u = np.array([0.0, n[2], -n[1]], dtype=np.float64)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-6:
            u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            u /= u_norm
        v = np.cross(n, u)

        # Local coordinates relative to vertex i
        diffs = nbrs - vertices[i]
        x_loc = diffs @ u
        y_loc = diffs @ v
        z_loc = diffs @ n

        # Fit paraboloid z = 0.5 * (a*x^2 + 2*b*x*y + c*y^2)
        A = np.column_stack([0.5 * x_loc**2, x_loc * y_loc, 0.5 * y_loc**2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z_loc, rcond=None)
            a, b, c = coeffs[0], coeffs[1], coeffs[2]
            # Mean curvature H = 0.5 * (a + c), Gaussian curvature K = a*c - b^2
            H = 0.5 * (a + c)
            K = a * c - b**2
            mean_curvatures[i] = H
            gaussian_curvatures[i] = K
        except Exception:
            mean_curvatures[i] = 0.0
            gaussian_curvatures[i] = 0.0

    return mean_curvatures, gaussian_curvatures, normals


def refine_surface_normals_and_curvature(
    mesh_path: Union[str, Path],
    output_ply: Union[str, Path],
    output_obj: Optional[Union[str, Path]] = None,
    curvature_weight: float = 0.3
) -> Dict[str, Any]:
    """
    Curvature-Guided Normal Refinement:
    Applies bilateral normal smoothing weighted by local surface curvature:
    - High-curvature anatomical features (dental cusps, incisal edges, marginal ridges)
      are preserved sharply.
    - Low-curvature regions (buccal/lingual smooth enamel faces) are smoothed to eliminate
      Marching Cubes discretization artifacts.
    """
    path = Path(mesh_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Surface mesh not found at {path}")

    mesh = trimesh.load_mesh(str(path), process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    # Compute vertex normals and curvatures
    H, K, normals = compute_mesh_curvatures(mesh, k_neighbors=20)

    # Curvature-adaptive bilateral normal filtering
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    tree = cKDTree(vertices)
    _, nbr_indices = tree.query(vertices, k=15)

    refined_normals = np.zeros_like(normals)
    abs_H = np.abs(H)
    # Normalize curvature metric for weighting
    h_max = np.percentile(abs_H, 95) if len(abs_H) > 0 else 1.0
    h_norm = np.clip(abs_H / max(h_max, 1e-5), 0.0, 1.0)

    for i in range(len(vertices)):
        nbr_norms = normals[nbr_indices[i]]
        diffs = vertices[nbr_indices[i]] - vertices[i]
        dists = np.linalg.norm(diffs, axis=-1)

        # Spatial Gaussian weights
        spatial_weights = np.exp(-dists**2 / (2 * 0.02**2))

        # Range normal similarity weights
        norm_similarities = np.maximum(0.0, (nbr_norms @ normals[i]))
        range_weights = norm_similarities**2

        # Curvature preservation: high curvature vertices rely mostly on their own normal
        alpha = float(h_norm[i] * curvature_weight)
        weights = (1.0 - alpha) * (spatial_weights * range_weights)
        weights[0] += alpha * np.sum(weights)  # strengthen center vertex weight

        w_sum = np.sum(weights)
        if w_sum > 1e-6:
            filtered = (weights[:, None] * nbr_norms).sum(axis=0) / w_sum
            f_norm = np.linalg.norm(filtered)
            refined_normals[i] = filtered / max(f_norm, 1e-6)
        else:
            refined_normals[i] = normals[i]

    # Create refined mesh and point cloud
    mesh.vertex_normals = refined_normals

    out_p = Path(output_ply).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Export refined point cloud with oriented normals
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.ascontiguousarray(vertices, dtype=np.float64).copy())
    pcd_o3d.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(refined_normals, dtype=np.float64).copy())
    o3d.io.write_point_cloud(str(out_p), pcd_o3d)

    if output_obj:
        out_o = Path(output_obj).resolve()
        out_o.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_o))

    return {
        "vertices": int(len(vertices)),
        "mean_curvature_mean": float(np.mean(np.abs(H))),
        "mean_curvature_max": float(np.max(np.abs(H))),
        "gaussian_curvature_mean": float(np.mean(K)),
        "output_ply": str(out_p),
        "output_obj": str(output_obj) if output_obj else None
    }
