import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import warnings

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    import lpips
except ImportError:
    warnings.warn("Missing 2D metrics packages. Run: pip install scikit-image lpips")

try:
    import open3d as o3d
except ImportError:
    warnings.warn("Missing Open3D for 3D metrics. Run: pip install open3d")

def evaluate_2d(sample_dir):
    sample = Path(sample_dir)
    images_dir = sample / "images"
    renders_dir = sample / "gaussian_renders"
    
    if not renders_dir.exists():
        print("No renders found. Run render_gaussian.py first.")
        return {}

    psnr_vals, ssim_vals, lpips_vals = [], [], []
    loss_fn_vgg = lpips.LPIPS(net='vgg') if 'lpips' in globals() else None

    for gt_file in images_dir.glob("*.png"):
        render_file = renders_dir / gt_file.name
        if not render_file.exists():
            continue
            
        gt = np.array(Image.open(gt_file).convert("RGB"))
        pred = np.array(Image.open(render_file).convert("RGB"))

        psnr_vals.append(peak_signal_noise_ratio(gt, pred))
        ssim_vals.append(structural_similarity(gt, pred, channel_axis=-1))

        if loss_fn_vgg:
            gt_t = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0
            pred_t = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0
            lpips_vals.append(loss_fn_vgg(gt_t, pred_t).item())

    return {
        "PSNR": np.mean(psnr_vals) if psnr_vals else 0,
        "SSIM": np.mean(ssim_vals) if ssim_vals else 0,
        "LPIPS": np.mean(lpips_vals) if lpips_vals else 0
    }

def evaluate_3d(sample_dir):
    if 'o3d' not in globals():
        return {}
        
    sample = Path(sample_dir)
    gt_mesh_path = sample / "mesh_normalized.obj"
    pred_ply_path = sample / "registered_reconstruction.ply"

    if not gt_mesh_path.exists() or not pred_ply_path.exists():
        print("Missing 3D models for evaluation.")
        return {}

    gt_mesh = o3d.io.read_triangle_mesh(str(gt_mesh_path))
    pred_pcd = o3d.io.read_point_cloud(str(pred_ply_path))

    # Sample points from GT mesh
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=len(pred_pcd.points))
    
    dist_pred_to_gt = pred_pcd.compute_point_cloud_distance(gt_pcd)
    dist_gt_to_pred = gt_pcd.compute_point_cloud_distance(pred_pcd)

    cd = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)
    
    threshold = 0.04
    precision = np.sum(np.asarray(dist_pred_to_gt) < threshold) / len(dist_pred_to_gt)
    recall = np.sum(np.asarray(dist_gt_to_pred) < threshold) / len(dist_gt_to_pred)
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Chamfer_Distance": cd,
        "F1_Score_0.04": f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    args = parser.parse_args()

    print("Evaluating 2D Metrics (Stage 4)...")
    res_2d = evaluate_2d(args.sample)
    for k, v in res_2d.items():
        print(f"{k}: {v:.4f}")

    print("\nEvaluating 3D Metrics (Stage 4)...")
    res_3d = evaluate_3d(args.sample)
    for k, v in res_3d.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
