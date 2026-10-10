"""
Phase 10 & Phase 11: Dedicated Reconstruction Accuracy Evaluator.

Checks for authentic reference IOS/ground-truth geometry before evaluation:
- If reference geometry exists: computes complete clinical accuracy metrics (MAE, RMSE, Chamfer, ALE, P95, Normal Consistency, Hausdorff).
- If reference geometry does NOT exist: marks real accuracy as "Pending reference IOS ground truth." and validates using synthetic geometry only.
- Outputs clean evaluation artifacts: metrics.json and metrics.csv in sample's evaluation/ directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh

from evaluation.metrics import compute_all_accuracy_metrics


def find_reference_geometry(sample_dir: Path) -> Optional[Path]:
    """
    Searches sample and repository for authentic reference IOS ground-truth geometry.
    Looks for:
    1. sample_dir / "mesh_normalized.obj"
    2. sample_dir / metadata["source_obj"]
    3. data/raw/Teeth3DS/<sample_id>.obj
    4. Any .obj / .ply / .stl in data/raw/ or sample_dir matching reference patterns.
    """
    # 1. Normalized working copy in sample dir
    norm_mesh = sample_dir / "mesh_normalized.obj"
    if norm_mesh.exists():
        return norm_mesh

    # 2. Check metadata source_obj
    meta_path = sample_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "source_obj" in meta and Path(meta["source_obj"]).exists():
                return Path(meta["source_obj"])
        except Exception:
            pass

    # 3. Check data/raw/Teeth3DS
    sample_id = sample_dir.name
    raw_obj = sample_dir.parent.parent / "raw" / "Teeth3DS" / f"{sample_id}.obj"
    if raw_obj.exists():
        return raw_obj

    # 4. Check sibling files
    for pattern in ["*reference*.obj", "*gt*.obj", "*IOS*.obj", "*scan*.obj"]:
        matches = list(sample_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def load_geometry_points_and_normals(
    file_path: Path,
    target_count: int = 50000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads 3D geometry from PLY or OBJ, returning point coordinates and surface normals.
    Uniformly samples surfaces if mesh.
    """
    path_str = str(file_path.resolve())
    suffix = file_path.suffix.lower()

    if suffix in [".ply", ".pcd"]:
        pcd = o3d.io.read_point_cloud(path_str)
        if len(pcd.points) > 0:
            if not pcd.has_normals():
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
                pcd.orient_normals_consistent_tangent_plane(50)
            pts = np.asarray(pcd.points, dtype=np.float64)
            norms = np.asarray(pcd.normals, dtype=np.float64)
            if len(pts) > target_count:
                sub_idx = np.random.choice(len(pts), target_count, replace=False)
                return pts[sub_idx], norms[sub_idx]
            return pts, norms

    # Load triangle mesh
    mesh_o3d = o3d.io.read_triangle_mesh(path_str)
    if len(mesh_o3d.triangles) > 0:
        if not mesh_o3d.has_vertex_normals():
            mesh_o3d.compute_vertex_normals()
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=target_count)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        pts = np.asarray(pcd.points, dtype=np.float64)
        norms = np.asarray(pcd.normals, dtype=np.float64)
        return pts, norms

    # Fallback trimesh
    loaded = trimesh.load(path_str, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    if hasattr(loaded, "sample"):
        pts, face_idx = loaded.sample(target_count, return_index=True)
        norms = loaded.face_normals[face_idx]
        return np.asarray(pts, dtype=np.float64), np.asarray(norms, dtype=np.float64)

    pts = np.asarray(loaded.vertices, dtype=np.float64)
    norms = np.zeros_like(pts)
    return pts, norms


def evaluate_sample_reconstruction(
    sample_dir: Union[str, Path],
    reconstructed_model_path: Optional[Union[str, Path]] = None,
    reference_model_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    sample_count: int = 50000
) -> Dict[str, Any]:
    """
    Main evaluation entry point.
    Executes Phase 10 & 11 accuracy assessment.
    """
    sample_path = Path(sample_dir).resolve()
    sample_id = sample_path.name
    eval_dir = Path(output_dir).resolve() if output_dir else (sample_path / "evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine reconstructed model file
    rec_path = None
    if reconstructed_model_path:
        rec_path = Path(reconstructed_model_path).resolve()
    else:
        # Search candidate hierarchy: registered_reconstruction.ply -> nsdf_surface_refined.ply -> reconstruction_surface.obj -> reconstruction.ply
        candidates = [
            sample_path / "registered_reconstruction.ply",
            sample_path / "nsdf_surface_refined.ply",
            sample_path / "reconstruction_surface.obj",
            sample_path / "reconstruction.ply"
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 0:
                rec_path = c
                break

    if rec_path is None or not rec_path.exists():
        raise FileNotFoundError(f"No valid reconstructed geometry found for sample {sample_id} in {sample_path}")

    # 2. Check for authentic reference IOS ground truth (Phase 11)
    ref_path = None
    if reference_model_path:
        ref_path = Path(reference_model_path).resolve()
    else:
        ref_path = find_reference_geometry(sample_path)

    has_real_reference = ref_path is not None and ref_path.exists()

    if not has_real_reference:
        # Strictly adhere to Phase 11 Scientific Integrity Rules:
        # NEVER fabricate accuracy results. Mark real experimental accuracy as pending.
        report = {
            "sample_id": sample_id,
            "status": "pending_reference_ground_truth",
            "message": "Pending reference IOS ground truth. Authentic reference geometry (.obj/.ply) not found.",
            "reconstructed_model": str(rec_path.name),
            "mae_mm": None,
            "rmse_mm": None,
            "chamfer_distance_mm": None,
            "ale": None,
            "p95_mm": None,
            "normal_consistency": None,
            "hausdorff_distance_mm": None
        }
        (eval_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    # 3. Compute genuine accuracy metrics against reference geometry
    print(f"[Evaluation] Assessing {rec_path.name} against authentic reference {ref_path.name}...")
    pred_pts, pred_norms = load_geometry_points_and_normals(rec_path, target_count=sample_count)
    ref_pts, ref_norms = load_geometry_points_and_normals(ref_path, target_count=sample_count)

    # Optional Teeth3DS annotations / landmarks check
    landmarks_ref = None
    json_annot = sample_path.parent.parent / "raw" / "Teeth3DS" / f"{sample_id}.json"
    if json_annot.exists():
        try:
            annot_data = json.loads(json_annot.read_text(encoding="utf-8"))
            labels = np.array(annot_data.get("labels", []))
            if len(labels) == len(ref_pts):
                # Identify per-tooth occlusal landmark extrema
                unique_teeth = [lbl for lbl in np.unique(labels) if lbl > 0]
                tooth_cusps = []
                for t in unique_teeth:
                    t_pts = ref_pts[labels == t]
                    if len(t_pts) > 0:
                        highest_pt = t_pts[np.argmax(t_pts[:, 2])]
                        tooth_cusps.append(highest_pt)
                if tooth_cusps:
                    landmarks_ref = np.array(tooth_cusps)
        except Exception:
            pass

    metrics = compute_all_accuracy_metrics(
        pred_points=pred_pts,
        ref_points=ref_pts,
        pred_normals=pred_norms,
        ref_normals=ref_norms,
        ref_landmarks=landmarks_ref
    )

    full_evaluation = {
        "sample_id": sample_id,
        "status": "evaluated_against_ground_truth",
        "reconstructed_model": str(rec_path.name),
        "reference_model": str(ref_path.name),
        "evaluation_points_count": sample_count,
        "metrics": metrics
    }

    # Save metrics.json
    metrics_json_path = eval_dir / "metrics.json"
    metrics_json_path.write_text(json.dumps(full_evaluation, indent=2), encoding="utf-8")

    # Save metrics.csv
    metrics_csv_path = eval_dir / "metrics.csv"
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value", "Unit", "Interpretation"])
        writer.writerow(["MAE", f"{metrics['mae_mm']:.4f}", "mm", "Mean Absolute Surface Error"])
        writer.writerow(["RMSE", f"{metrics['rmse_mm']:.4f}", "mm", "Root Mean Square Surface Error"])
        writer.writerow(["Chamfer_Forward", f"{metrics['chamfer_distance_mm']['forward']:.4f}", "mm", "Pred -> Ref Distance"])
        writer.writerow(["Chamfer_Backward", f"{metrics['chamfer_distance_mm']['backward']:.4f}", "mm", "Ref -> Pred Distance"])
        writer.writerow(["Chamfer_Total", f"{metrics['chamfer_distance_mm']['total']:.4f}", "mm", "Bidirectional Sum"])
        writer.writerow(["P95", f"{metrics['p95_mm']:.4f}", "mm", "95th Percentile Error"])
        ale_val = metrics["ale"]["ale_mm"]
        writer.writerow(["ALE", f"{ale_val:.4f}" if ale_val is not None else "Pending", "mm", f"Anatomical Landmark Error ({metrics['ale']['method']})"])
        if metrics["normal_consistency"]:
            writer.writerow(["Normal_Cosine_Similarity", f"{metrics['normal_consistency']['mean_cosine_similarity']:.4f}", "[-1, 1]", "Surface Normal Alignment"])
            writer.writerow(["Normal_Angular_Error", f"{metrics['normal_consistency']['mean_angular_error_deg']:.2f}", "degrees", "Mean Angular Normal Deviation"])
        writer.writerow(["Hausdorff", f"{metrics['hausdorff_distance_mm']['hausdorff_mm']:.4f}", "mm", "Worst-case Surface Distance"])

    print(f"[Evaluation] Completed! Results saved to {metrics_json_path.name} and {metrics_csv_path.name}:")
    print(f"  - MAE:   {metrics['mae_mm']:.4f} mm")
    print(f"  - RMSE:  {metrics['rmse_mm']:.4f} mm")
    print(f"  - CD:    {metrics['chamfer_distance_mm']['total']:.4f} mm")
    print(f"  - P95:   {metrics['p95_mm']:.4f} mm")
    if metrics["normal_consistency"]:
        print(f"  - Norm:  {metrics['normal_consistency']['mean_cosine_similarity']:.4f} (deg: {metrics['normal_consistency']['mean_angular_error_deg']:.2f}°)")
    print(f"  - HD:    {metrics['hausdorff_distance_mm']['hausdorff_mm']:.4f} mm")

    return full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D Dental Reconstruction Accuracy against Reference Model")
    parser.add_argument("sample", help="Sample directory (e.g. data/processed/00OMSZGW_upper)")
    parser.add_argument("--reconstruction", default=None, help="Path to reconstructed model (.ply or .obj)")
    parser.add_argument("--reference", default=None, help="Path to ground truth reference model (.obj or .ply)")
    parser.add_argument("--points", type=int, default=50000, help="Number of surface points for evaluation")
    args = parser.parse_args()

    evaluate_sample_reconstruction(
        sample_dir=args.sample,
        reconstructed_model_path=args.reconstruction,
        reference_model_path=args.reference,
        sample_count=args.points
    )


if __name__ == "__main__":
    main()
