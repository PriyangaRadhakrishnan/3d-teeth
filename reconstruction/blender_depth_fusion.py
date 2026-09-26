"""Fuse rendered depth maps into a dense, high-accuracy registered 3D point cloud."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

VIEW_ALIASES = {
    "frontal": ["frontal", "front"],
    "left_buccal": ["left_buccal", "left"],
    "right_buccal": ["right_buccal", "right"],
    "maxillary_occlusal": ["maxillary_occlusal", "top"],
    "mandibular_occlusal": ["mandibular_occlusal", "back"]
}

def resolve_file(sample_dir, subfolder, view_name, extensions):
    aliases = VIEW_ALIASES.get(view_name, [view_name])
    for alias in aliases:
        for ext in extensions:
            target = sample_dir / subfolder / f"{alias}{ext}"
            if target.exists():
                return target
            matches = list((sample_dir / subfolder).glob(f"{alias}*{ext}"))
            if matches:
                return matches[0]
    return None

def load_depth_map(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    # Invert visualization depth map to metric scale range (0 to 2.5 units)
    metric_depth = (1.0 - arr) * 2.5
    return img.width, img.height, metric_depth

def load_mask(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0

def load_color(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)

def rotation_matrix_from_euler(roll, pitch, yaw):
    cr = math.cos(roll); sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw); sy = math.sin(yaw)
    
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def fuse(sample_dir, output_path, stride=1):
    sample_dir = Path(sample_dir)
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    cameras = json.loads((sample_dir / metadata["camera_file"]).read_text(encoding="utf-8"))
    points = []
    colors = []
    
    views = metadata.get("views", ["frontal", "left_buccal", "right_buccal", "maxillary_occlusal", "mandibular_occlusal"])

    for view in views:
        depth_path = resolve_file(sample_dir, "depth_vis", view, [".png"])
        mask_path = resolve_file(sample_dir, "masks", view, [".png"])
        image_path = resolve_file(sample_dir, "images", view, [".png"])

        if not depth_path or not depth_path.exists():
            continue

        width, height, depth_pixels = load_depth_map(depth_path)
        mask_pixels = load_mask(mask_path) if mask_path and mask_path.exists() else np.ones((height, width))
        color_pixels = load_color(image_path) if image_path and image_path.exists() else np.full((height, width, 3), 220, dtype=np.uint8)

        camera_key = view if view in cameras else VIEW_ALIASES.get(view, [view])[0]
        if camera_key not in cameras:
            camera_key = next((k for k in cameras if any(a in k for a in VIEW_ALIASES.get(view, []))), list(cameras.keys())[0])

        camera = cameras[camera_key]
        focal = camera["focal_length_pixels"]
        cx, cy = camera["principal_point"]
        
        pos = np.array(camera["position"])
        rot_euler = camera["rotation_euler"]
        R = rotation_matrix_from_euler(*rot_euler)

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                if mask_pixels[y, x] < 0.15:
                    continue
                depth = float(depth_pixels[y, x])
                if depth <= 0.05 or depth > 4.0:
                    continue

                # Camera coordinate back-projection
                cam_x = (x - cx) * depth / focal
                cam_y = (y - cy) * depth / focal
                cam_z = -depth
                
                pt_cam = np.array([cam_x, cam_y, cam_z])
                pt_world = R @ pt_cam + pos
                
                points.append(pt_world)
                colors.append(color_pixels[y, x])

    if not points:
        raise RuntimeError("Depth fusion generated zero valid points")

    cloud = trimesh.PointCloud(np.array(points), colors=np.array(colors))
    cloud.export(output_path)
    print(f"Exported dense point cloud to {output_path} ({len(points)} points)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    fuse(args.sample, args.output, args.stride)

if __name__ == "__main__":
    main()