from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(Path(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise ValueError(f"Unable to load a non-empty mesh from {path}")
    return loaded


def get_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(mesh.vertices)


def get_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(mesh.faces)


def get_bounds(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    return mesh.bounds[0].copy(), mesh.bounds[1].copy()


def get_center(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.bounds.mean(axis=0)


def get_scale(mesh: trimesh.Trimesh) -> float:
    return float(np.max(mesh.extents))


def inspect_obj(obj_path: str | Path, annotation_path: str | Path | None = None) -> dict[str, Any]:
    mesh = load_mesh(obj_path)
    minimum, maximum = get_bounds(mesh)
    result: dict[str, Any] = {
        "path": str(Path(obj_path)),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_min": minimum.tolist(),
        "bounds_max": maximum.tolist(),
        "center": get_center(mesh).tolist(),
        "extents": mesh.extents.tolist(),
        "scale": get_scale(mesh),
        "is_watertight": bool(mesh.is_watertight),
    }
    if annotation_path and Path(annotation_path).exists():
        annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
        labels = annotation.get("labels", [])
        instances = annotation.get("instances", [])
        result["annotation"] = {
            "jaw": annotation.get("jaw"),
            "patient_id": annotation.get("id_patient"),
            "label_count": len(labels),
            "instance_count": len(instances),
            "labels_match_vertices": len(labels) == len(mesh.vertices),
            "instances_match_vertices": len(instances) == len(mesh.vertices),
        }
    return result