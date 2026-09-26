from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .obj_loader import load_mesh


def normalize_mesh(source_obj: str | Path, output_dir: str | Path, target_extent: float = 2.0) -> dict:
    mesh = load_mesh(source_obj)
    mesh.remove_infinite_values()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    original_center = mesh.bounds.mean(axis=0)
    original_scale = float(np.max(mesh.extents))
    if original_scale <= 0:
        raise ValueError("Mesh has zero extent")
    scale = target_extent / original_scale
    mesh.apply_translation(-original_center)
    mesh.apply_scale(scale)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized_obj = output / "mesh_normalized.obj"
    mesh.export(normalized_obj)
    metadata = {
        "source_obj": str(Path(source_obj).resolve()),
        "original_center": original_center.tolist(),
        "scale": scale,
        "translation": (-original_center).tolist(),
        "rotation": np.eye(3).tolist(),
        "normalized_bounds": mesh.bounds.tolist(),
        "working_extent": target_extent,
        "normalized_obj": normalized_obj.name,
    }
    (output / "mesh_transform.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata