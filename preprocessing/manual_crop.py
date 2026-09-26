from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .obj_loader import load_mesh


def crop_upper_surface(source_obj: str | Path, output_dir: str | Path, z_fraction: float = 0.58) -> Path:
    source = Path(source_obj)
    mesh = load_mesh(source)
    minimum, maximum = mesh.bounds
    cutoff = float(minimum[2] + (maximum[2] - minimum[2]) * z_fraction)
    keep_vertices = np.asarray(mesh.vertices)[:, 2] >= cutoff
    face_keep = keep_vertices[np.asarray(mesh.faces)].all(axis=1)
    if int(face_keep.sum()) < 10:
        raise ValueError("Manual crop retained too few faces; lower z_fraction")
    cropped = mesh.submesh([np.flatnonzero(face_keep)], append=True, repair=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cropped_path = output / f"{source.stem}_manual_crown.obj"
    cropped.export(cropped_path)
    metadata = {
        "source_obj": str(source.resolve()),
        "method": "manual_upper_surface_crop",
        "z_fraction": z_fraction,
        "z_cutoff": cutoff,
        "source_bounds": mesh.bounds.tolist(),
        "cropped_bounds": cropped.bounds.tolist(),
        "source_vertices": int(len(mesh.vertices)),
        "cropped_vertices": int(len(cropped.vertices)),
        "cropped_faces": int(len(cropped.faces)),
        "warning": "Manual crop for prototype visualization; not a validated Teeth3DS crown annotation.",
    }
    (output / "manual_crop.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return cropped_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a separate manual crown crop from a Teeth3DS OBJ")
    parser.add_argument("obj")
    parser.add_argument("--output-dir", default="data/raw/Teeth3DS/manual_crop")
    parser.add_argument("--z-fraction", type=float, default=0.58)
    args = parser.parse_args()
    print(crop_upper_surface(args.obj, args.output_dir, args.z_fraction))


if __name__ == "__main__":
    main()