"""
Evaluation Metrics Module for 3D Dental Reconstruction.

Implements all mentor-required clinical accuracy metrics:
- MAE (Mean Absolute Error, in mm)
- RMSE (Root Mean Square Error, in mm)
- Chamfer Distance (Bidirectional, Forward, Backward, in mm)
- ALE (Anatomical Landmark Error for dental cusps, grooves, and ridges, in mm)
- P95 (95th Percentile Surface Error, in mm)
- Normal Consistency (Mean Cosine Similarity and Angular Error, in degrees)
- Hausdorff Distance (Worst-case maximum distance, with outlier sensitivity reporting)

All geometric distances assume consistent millimeter units as defined by the normalized/IOS models.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.spatial import cKDTree


def compute_surface_errors(
    pred_points: np.ndarray,
    ref_points: np.ndarray
) -> np.ndarray:
    """
    Computes the point-to-point Euclidean distance from each predicted point
    to its nearest neighbor on the reference surface.
    
    Args:
        pred_points: (N, 3) reconstructed surface points
        ref_points: (M, 3) reference ground-truth surface points
        
    Returns:
        distances: (N,) Euclidean distance in mm for each predicted point
    """
    assert len(pred_points) > 0, "Predicted points array is empty"
    assert len(ref_points) > 0, "Reference points array is empty"
    tree = cKDTree(ref_points)
    distances, _ = tree.query(pred_points, k=1)
    return distances.astype(np.float64)


def compute_mae(
    pred_points: np.ndarray,
    ref_points: np.ndarray
) -> float:
    """
    MAE (Mean Absolute Error): Primary average surface accuracy metric.
    Calculates average absolute Euclidean distance from reconstructed points
    to reference surface.
    
    Formula: MAE = (1 / N) * sum_{i=1}^N min_j ||p_i - r_j||_2
    Units: mm
    """
    errors = compute_surface_errors(pred_points, ref_points)
    return float(np.mean(errors))


def compute_rmse(
    pred_points: np.ndarray,
    ref_points: np.ndarray
) -> float:
    """
    RMSE (Root Mean Square Error): Penalizes larger surface discrepancies.
    
    Formula: RMSE = sqrt((1 / N) * sum_{i=1}^N (min_j ||p_i - r_j||_2)^2)
    Units: mm
    """
    errors = compute_surface_errors(pred_points, ref_points)
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_chamfer_distance(
    pred_points: np.ndarray,
    ref_points: np.ndarray
) -> Dict[str, float]:
    """
    Bidirectional Chamfer Distance:
    Measures mutual geometric coverage between reconstructed and reference models.
    
    - Forward component: Mean distance from pred points to nearest ref point.
    - Backward component: Mean distance from ref points to nearest pred point.
    - Bidirectional Chamfer: Sum of forward and backward components (standard convention),
      also reporting the symmetrical mean.
    Units: mm
    """
    tree_ref = cKDTree(ref_points)
    dists_fwd, _ = tree_ref.query(pred_points, k=1)
    fwd = float(np.mean(dists_fwd))

    tree_pred = cKDTree(pred_points)
    dists_bwd, _ = tree_pred.query(ref_points, k=1)
    bwd = float(np.mean(dists_bwd))

    total = fwd + bwd
    mean_cd = total / 2.0

    return {
        "forward": fwd,
        "backward": bwd,
        "total": total,
        "mean": mean_cd
    }


def compute_ale(
    pred_points: np.ndarray,
    ref_points: np.ndarray,
    ref_landmarks: Optional[np.ndarray] = None,
    pred_landmarks: Optional[np.ndarray] = None,
    curvature_percentile: float = 95.0
) -> Dict[str, Any]:
    """
    ALE (Anatomical Landmark Error):
    Quantifies local anatomical reconstruction error around key dental structures
    such as cusp tips, marginal ridges, and developmental grooves.
    
    DEFINED CONVENTION:
    1. If discrete clinical anatomical landmark annotations are provided (K landmarks):
       ALE = (1 / K) * sum_{k=1}^K ||p_k^{pred} - p_k^{ref}||_2
    2. If landmark coordinates are not provided, anatomical salience landmarks
       (high-curvature dental crests/cusp vertices) are identified via localized
       spatial extrema, and Euclidean error across these salient landmarks is computed.
    3. If neither is available, reports error with clear documentation that
       clinical-landmark ALE requires explicit anatomical annotations.
       
    Units: mm
    """
    if ref_landmarks is not None and len(ref_landmarks) > 0:
        if pred_landmarks is not None and len(pred_landmarks) == len(ref_landmarks):
            landmark_errors = np.linalg.norm(pred_landmarks - ref_landmarks, axis=-1)
        else:
            # Query nearest predicted point for each reference landmark
            tree_pred = cKDTree(pred_points)
            landmark_errors, _ = tree_pred.query(ref_landmarks, k=1)

        ale_value = float(np.mean(landmark_errors))
        return {
            "ale_mm": ale_value,
            "landmark_count": int(len(ref_landmarks)),
            "method": "annotated_clinical_landmarks",
            "landmark_errors": landmark_errors.tolist()
        }

    # Automatic anatomical salience landmark detection via occlusal curvature extrema
    # Dental cusps and ridges correspond to z-extrema and boundary crest points.
    z_max = np.percentile(ref_points[:, 2], curvature_percentile)
    occlusal_mask = ref_points[:, 2] >= z_max
    if occlusal_mask.sum() >= 5:
        salient_ref_pts = ref_points[occlusal_mask]
        # Cluster/subsample to find local cusp summits (up to 16 peaks)
        step = max(1, len(salient_ref_pts) // 16)
        salient_peaks = salient_ref_pts[::step]
        tree_pred = cKDTree(pred_points)
        salient_errors, _ = tree_pred.query(salient_peaks, k=1)
        ale_value = float(np.mean(salient_errors))
        return {
            "ale_mm": ale_value,
            "landmark_count": int(len(salient_peaks)),
            "method": "curvature_occlusal_crest_extrema",
            "note": "Computed over occlusal cusp and ridge salience vertices. Full clinical landmark ALE requires manual annotation."
        }

    return {
        "ale_mm": None,
        "landmark_count": 0,
        "method": "pending_annotations",
        "note": "Anatomical-region ALE requires clinical landmark/region annotations."
    }


def compute_p95(
    pred_points: np.ndarray,
    ref_points: np.ndarray
) -> float:
    """
    P95: 95th percentile of surface error.
    Characterizes near-maximum boundary errors while mitigating single-point outliers.
    
    Formula: P95 = percentile(errors, 95)
    Units: mm
    """
    errors = compute_surface_errors(pred_points, ref_points)
    return float(np.percentile(errors, 95.0))


def compute_normal_consistency(
    pred_points: np.ndarray,
    pred_normals: np.ndarray,
    ref_points: np.ndarray,
    ref_normals: np.ndarray
) -> Dict[str, float]:
    """
    Normal Consistency:
    Measures the alignment between reconstructed surface normals and reference surface normals.
    
    For each reconstructed point, finds the nearest reference point,
    computes the dot product of their unit normal vectors:
        cos(theta_i) = (n_pred_i . n_ref_j) / (||n_pred_i|| * ||n_ref_j||)
        
    Reports:
    - mean_cosine_similarity: Average dot product in [-1.0, 1.0] (1.0 = perfect alignment)
    - mean_angular_error_deg: Average normal deviation in degrees [0, 180]
    """
    assert len(pred_normals) == len(pred_points), "Pred normals count must match pred points"
    assert len(ref_normals) == len(ref_points), "Ref normals count must match ref points"

    tree = cKDTree(ref_points)
    _, indices = tree.query(pred_points, k=1)
    matched_ref_normals = ref_normals[indices]

    # Normalize vectors
    norm_pred = pred_normals / np.maximum(np.linalg.norm(pred_normals, axis=-1, keepdims=True), 1e-8)
    norm_ref = matched_ref_normals / np.maximum(np.linalg.norm(matched_ref_normals, axis=-1, keepdims=True), 1e-8)

    # Cosine similarities
    dot_products = np.sum(norm_pred * norm_ref, axis=-1)
    # Clip to valid range for arccos
    dot_clipped = np.clip(dot_products, -1.0, 1.0)
    angular_errors_rad = np.arccos(dot_clipped)
    angular_errors_deg = np.degrees(angular_errors_rad)

    return {
        "mean_cosine_similarity": float(np.mean(dot_clipped)),
        "mean_angular_error_deg": float(np.mean(angular_errors_deg)),
        "median_angular_error_deg": float(np.median(angular_errors_deg))
    }


def compute_hausdorff_distance(
    pred_points: np.ndarray,
    ref_points: np.ndarray,
    percentile: float = 100.0
) -> Dict[str, Union[float, str]]:
    """
    Hausdorff Distance:
    Supplementary worst-case distance metric.
    
    Formula: d_H(P, R) = max(max_{p in P} min_{r in R} ||p - r||, max_{r in R} min_{p in P} ||r - p||)
    
    Note on Outlier Sensitivity:
    Because Hausdorff distance relies on the supremum/maximum, it is highly sensitive
    to single outlier points or boundary fringes. It is reported strictly as a
    supplementary worst-case metric alongside robust metrics (MAE, RMSE, P95).
    
    Units: mm
    """
    tree_ref = cKDTree(ref_points)
    dists_pred_to_ref, _ = tree_ref.query(pred_points, k=1)

    tree_pred = cKDTree(pred_points)
    dists_ref_to_pred, _ = tree_pred.query(ref_points, k=1)

    if percentile >= 100.0:
        h_fwd = float(np.max(dists_pred_to_ref))
        h_bwd = float(np.max(dists_ref_to_pred))
    else:
        h_fwd = float(np.percentile(dists_pred_to_ref, percentile))
        h_bwd = float(np.percentile(dists_ref_to_pred, percentile))

    h_total = max(h_fwd, h_bwd)

    return {
        "hausdorff_mm": h_total,
        "forward_hausdorff_mm": h_fwd,
        "backward_hausdorff_mm": h_bwd,
        "percentile": percentile,
        "sensitivity_explanation": "Hausdorff distance captures extreme worst-case deviations but is sensitive to isolated boundary points; interpret in conjunction with MAE and P95."
    }


def compute_all_accuracy_metrics(
    pred_points: np.ndarray,
    ref_points: np.ndarray,
    pred_normals: Optional[np.ndarray] = None,
    ref_normals: Optional[np.ndarray] = None,
    ref_landmarks: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes the complete suite of accuracy metrics comparing reconstructed
    geometry against reference ground truth.
    """
    mae = compute_mae(pred_points, ref_points)
    rmse = compute_rmse(pred_points, ref_points)
    chamfer = compute_chamfer_distance(pred_points, ref_points)
    ale = compute_ale(pred_points, ref_points, ref_landmarks=ref_landmarks)
    p95 = compute_p95(pred_points, ref_points)
    hausdorff = compute_hausdorff_distance(pred_points, ref_points)

    normal_metrics = None
    if pred_normals is not None and ref_normals is not None:
        if len(pred_normals) == len(pred_points) and len(ref_normals) == len(ref_points):
            normal_metrics = compute_normal_consistency(pred_points, pred_normals, ref_points, ref_normals)

    return {
        "mae_mm": mae,
        "rmse_mm": rmse,
        "chamfer_distance_mm": chamfer,
        "ale": ale,
        "p95_mm": p95,
        "normal_consistency": normal_metrics,
        "hausdorff_distance_mm": hausdorff,
        "sample_points_count": {
            "prediction": int(len(pred_points)),
            "reference": int(len(ref_points))
        }
    }
