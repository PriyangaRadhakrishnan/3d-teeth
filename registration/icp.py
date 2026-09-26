from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_point_cloud(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    loaded = trimesh.load(Path(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    points = np.asarray(loaded.vertices, dtype=np.float64)
    colors = None
    if loaded.visual.kind == "vertex" and loaded.visual.vertex_colors is not None:
        colors = np.asarray(loaded.visual.vertex_colors[:, :3])
    return points, colors


def sample_mesh(path: str | Path, count: int, seed: int = 0) -> np.ndarray:
    mesh = trimesh.load_mesh(Path(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    return np.asarray(mesh.sample(min(count, max(len(mesh.vertices), count)), seed=seed), dtype=np.float64)


def best_fit_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def run_icp(source: np.ndarray, target: np.ndarray, iterations: int = 30, threshold: float | None = None) -> tuple[np.ndarray, dict]:
    rotation = np.eye(3)
    translation = np.zeros(3)
    tree = cKDTree(target)
    distances = np.full(len(source), np.inf)
    for _ in range(iterations):
        transformed = source @ rotation.T + translation
        distances, indices = tree.query(transformed)
        keep = distances <= threshold if threshold is not None else np.ones(len(source), dtype=bool)
        if keep.sum() < 3:
            raise RuntimeError("ICP retained fewer than three correspondences")
        delta_rotation, delta_translation = best_fit_transform(transformed[keep], target[indices[keep]])
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
        if np.linalg.norm(delta_translation) < 1e-7 and np.linalg.norm(delta_rotation - np.eye(3)) < 1e-7:
            break
    transformed = source @ rotation.T + translation
    distances, _ = tree.query(transformed)
    inliers = distances <= threshold if threshold is not None else np.ones(len(source), dtype=bool)
    result = {"fitness": float(inliers.mean()), "inlier_rmse": float(np.sqrt(np.mean(distances[inliers] ** 2))), "correspondences": int(inliers.sum()), "iterations": iterations, "transformation": np.vstack([np.column_stack([rotation, translation]), [0, 0, 0, 1]]).tolist()}
    return transformed, result


def register(reconstruction_path: str | Path, original_obj: str | Path, transform_path: str | Path, output_path: str | Path, sample_count: int = 100000) -> dict:
    source, colors = load_point_cloud(reconstruction_path)
    target = sample_mesh(original_obj, sample_count)
    
    # Scale source point cloud to match target bounding box if target is normalized
    target_extent = np.linalg.norm(target.max(axis=0) - target.min(axis=0))
    source_extent = np.linalg.norm(source.max(axis=0) - source.min(axis=0))
    if source_extent > 0 and target_extent > 0:
        scale_ratio = target_extent / source_extent
        if abs(scale_ratio - 1.0) > 0.1:
            source = source * scale_ratio
            
    registered, result = run_icp(source, target)
    cloud = trimesh.PointCloud(registered, colors=colors)
    cloud.export(output_path)
    result.update({"reconstruction": str(Path(reconstruction_path)), "original_obj": str(Path(original_obj)), "output": str(Path(output_path))})
    Path(output_path).with_name("registration_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a reconstruction point cloud against a Teeth3DS OBJ")
    parser.add_argument("reconstruction")
    parser.add_argument("original_obj")
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output", default="registered_reconstruction.ply")
    parser.add_argument("--sample-count", type=int, default=50000)
    args = parser.parse_args()
    print(json.dumps(register(args.reconstruction, args.original_obj, args.transform, args.output, args.sample_count), indent=2))


if __name__ == "__main__":
    main()