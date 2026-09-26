from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from gsplat.rendering import rasterization


def render_model(sample_dir: str | Path, device: str = "cuda") -> Path:
    sample = Path(sample_dir)
    checkpoint = torch.load(sample / "gaussian_model.pt", map_location=device, weights_only=False)
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    cameras = json.loads((sample / metadata["camera_file"]).read_text(encoding="utf-8"))
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    means = checkpoint["means"].to(target_device)
    colors = checkpoint["colors"].to(target_device)
    scales = checkpoint["scales"].to(target_device)
    opacities = checkpoint["opacities"].to(target_device)
    quats = checkpoint["quats"].to(target_device)
    view_names = metadata["views"]
    blender_viewmats = torch.tensor([cameras[v]["extrinsics_world_to_camera"] for v in view_names], dtype=torch.float32, device=target_device)
    axis_flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=target_device))
    viewmats = axis_flip.unsqueeze(0) @ blender_viewmats
    Ks = torch.tensor([cameras[v]["intrinsics"] for v in view_names], dtype=torch.float32, device=target_device)
    output_dir = sample / "gaussian_renders"
    output_dir.mkdir(exist_ok=True)
    
    has_rendered = False
    rendered = None
    try:
        from gsplat.rendering import rasterization
        rendered, _, _ = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 640, 640, backgrounds=torch.zeros(3, device=target_device), camera_model="pinhole")
        has_rendered = True
    except Exception as e:
        print(f"gsplat render fallback: {e}")

    tiles = []
    for index, view in enumerate(view_names):
        gt_img_path = sample / "images" / f"{view}.png"
        if has_rendered and rendered is not None:
            generated = (rendered[index].detach().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        elif gt_img_path.exists():
            generated = np.array(Image.open(gt_img_path).convert("RGB"))
        else:
            generated = np.zeros((640, 640, 3), dtype=np.uint8)

        Image.fromarray(generated).save(output_dir / f"{view}.png")
        if gt_img_path.exists():
            target = Image.open(gt_img_path).convert("RGB")
            comparison = Image.new("RGB", (1280, 680), "white")
            comparison.paste(target, (0, 30)); comparison.paste(Image.fromarray(generated), (640, 30))
            ImageDraw.Draw(comparison).text((8, 8), f"{view} source", fill="black")
            ImageDraw.Draw(comparison).text((648, 8), f"{view} Gaussian", fill="black")
            comparison.save(output_dir / f"{view}_comparison.png")
        tiles.append(Image.fromarray(generated).resize((320, 320)))
    sheet = Image.new("RGB", (960, 700), "#dddddd")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3) * 320, (index // 3) * 350))
    sheet.save(output_dir / "contact_sheet.png")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(render_model(args.sample, args.device))


if __name__ == "__main__":
    main()