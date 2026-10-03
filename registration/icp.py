from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_point_cloud(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    p = Path(path)
    import open3d as o3d
    if p.suffix.lower() in [".ply", ".pcd"]:
        pcd = o3d.io.read_point_cloud(str(p))
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points, dtype=np.float64)
            normals = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
            return points, colors, normals

    loaded = trimesh.load(p, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    points = np.asarray(loaded.vertices, dtype=np.float64)
    colors = None
    if hasattr(loaded, "visual") and loaded.visual.kind == "vertex" and loaded.visual.vertex_colors is not None:
        colors = np.asarray(loaded.visual.vertex_colors[:, :3])
    normals = None
    if hasattr(loaded, "vertex_normals") and loaded.vertex_normals is not None and len(loaded.vertex_normals) == len(points):
        normals = np.asarray(loaded.vertex_normals, dtype=np.float64)
    return points, colors, normals


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
    result = {"fitness": float(inliers.mean()), "inlier_rmse": float(np.sqrt(np.mean(distances[inliers] ** 2))), "correspondences": int(inliers.sum()), "iterations": iterations, "transformation": np.vstack([np.column_stack([rotation, translation]), [0, 0, 0, 1]]).tolist(), "method": "point_to_point"}
    return transformed, result


def run_point_to_plane_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_normals: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
    iterations: int = 50,
    distance_threshold: float = 0.35,
) -> tuple[np.ndarray, dict]:
    """
    Performs Point-to-Plane ICP registration:
    Minimizes distance between source points and tangent planes at corresponding target points:
        min_{R, t} sum_i ((R * p_i + t - q_i) . n_i)^2
    """
    import open3d as o3d

    s_pcd = o3d.geometry.PointCloud()
    s_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(source_points, dtype=np.float64).copy())
    if source_normals is not None and len(source_normals) == len(source_points):
        s_pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(source_normals, dtype=np.float64).copy())
    else:
        s_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    t_pcd = o3d.geometry.PointCloud()
    t_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(target_points, dtype=np.float64).copy())
    if target_normals is not None and len(target_normals) == len(target_points):
        t_pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(target_normals, dtype=np.float64).copy())
    else:
        t_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    target_extent = float(np.linalg.norm(np.asarray(t_pcd.get_max_bound()) - np.asarray(t_pcd.get_min_bound())))

    # Coarse Point-to-Point initialization
    crit_coarse = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max(15, iterations // 3))
    res_coarse = o3d.pipelines.registration.registration_icp(
        s_pcd, t_pcd, max(target_extent * 0.5, 0.5), np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        crit_coarse
    )

    # Fine Point-to-Plane refinement
    crit_fine = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
    res_fine = o3d.pipelines.registration.registration_icp(
        s_pcd, t_pcd, distance_threshold, res_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        crit_fine
    )

    s_pcd.transform(res_fine.transformation)
    transformed_points = np.asarray(s_pcd.points)
    transformed_normals = np.asarray(s_pcd.normals) if s_pcd.has_normals() else None

    result = {
        "method": "point_to_plane",
        "fitness": float(res_fine.fitness),
        "inlier_rmse": float(res_fine.inlier_rmse),
        "correspondences": int(len(res_fine.correspondence_set)),
        "iterations": iterations,
        "transformation": res_fine.transformation.tolist(),
        "transformed_normals": transformed_normals.tolist() if transformed_normals is not None else None
    }
    return transformed_points, result


def register(
    reconstruction_path: str | Path,
    original_obj: str | Path,
    transform_path: str | Path,
    output_path: str | Path,
    sample_count: int = 100000,
    method: str = "point_to_plane"
) -> dict:
    source, colors, source_normals = load_point_cloud(reconstruction_path)
    
    # Load target mesh with normals
    import open3d as o3d
    target_mesh = o3d.io.read_triangle_mesh(str(Path(original_obj)))
    if not target_mesh.has_vertex_normals():
        target_mesh.compute_vertex_normals()
    target_pcd = target_mesh.sample_points_uniformly(number_of_points=min(sample_count, 100000))
    target = np.asarray(target_pcd.points, dtype=np.float64)
    target_normals = np.asarray(target_pcd.normals, dtype=np.float64)

    # Scale source point cloud to match target bounding box if target is normalized
    target_extent = np.linalg.norm(target.max(axis=0) - target.min(axis=0))
    source_extent = np.linalg.norm(source.max(axis=0) - source.min(axis=0))
    if source_extent > 0 and target_extent > 0:
        scale_ratio = target_extent / source_extent
        if abs(scale_ratio - 1.0) > 0.1:
            source = source * scale_ratio

    transformed_norms = None
    if method == "point_to_plane":
        try:
            registered, result = run_point_to_plane_icp(
                source, target,
                source_normals=source_normals,
                target_normals=target_normals,
                iterations=50
            )
            if result.get("transformed_normals") is not None:
                transformed_norms = np.asarray(result["transformed_normals"], dtype=np.float64)
        except Exception:
            registered, result = run_icp(source, target)
            result["method"] = "point_to_point_fallback"
    else:
        registered, result = run_icp(source, target)
        result["method"] = "point_to_point"

    # Export registered point cloud with normals using Open3D
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(registered, dtype=np.float64).copy())
    if transformed_norms is not None:
        out_pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(transformed_norms, dtype=np.float64).copy())
    else:
        out_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        out_pcd.orient_normals_consistent_tangent_plane(50)

    if colors is not None and len(colors) == len(registered):
        out_pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors[:, :3], dtype=np.float64) / 255.0)

    o3d.io.write_point_cloud(str(output_path), out_pcd)

    # Clean transformed_normals from json result dict if too large
    if "transformed_normals" in result:
        del result["transformed_normals"]

    result.update({
        "reconstruction": str(Path(reconstruction_path)),
        "original_obj": str(Path(original_obj)),
        "output": str(Path(output_path))
    })
    Path(output_path).with_name("registration_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a reconstruction point cloud against a Teeth3DS OBJ")
    parser.add_argument("reconstruction")
    parser.add_argument("original_obj")
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output", default="registered_reconstruction.ply")
    parser.add_argument("--sample-count", type=int, default=50000)
    parser.add_argument("--method", default="point_to_plane", choices=["point_to_plane", "point_to_point"])
    args = parser.parse_args()
    print(json.dumps(register(args.reconstruction, args.original_obj, args.transform, args.output, args.sample_count, method=args.method), indent=2))


if __name__ == "__main__":
    main()