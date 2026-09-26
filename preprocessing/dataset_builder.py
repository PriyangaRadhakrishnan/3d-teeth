from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .mesh_preprocessor import normalize_mesh
from .obj_loader import inspect_obj


def build_sample(obj_path: str | Path, output_root: str | Path, blender: str = "blender", resolution: int = 640) -> Path:
    obj = Path(obj_path).resolve()
    if not obj.exists() or not obj.is_file():
        raise FileNotFoundError(
            f"Teeth3DS OBJ not found: {obj}. Replace <sample>.obj with an actual OBJ filename."
        )
    sample_dir = Path(output_root).resolve() / obj.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    annotation = obj.with_suffix(".json")
    inspection = inspect_obj(obj, annotation if annotation.exists() else None)
    (sample_dir / "inspection.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    transform = normalize_mesh(obj, sample_dir)
    for name in ("images", "depth", "depth_vis", "normals", "masks", "cameras", "reports"):
        (sample_dir / name).mkdir(exist_ok=True)
    renderer = Path(__file__).with_name("blender_renderer.py").resolve()
    subprocess.run([blender, "-b", "-E", "BLENDER_EEVEE_NEXT", "--python", str(renderer), "--",
                    "--mesh", str(sample_dir / transform["normalized_obj"]), "--output", str(sample_dir),
                    "--resolution", str(resolution)], check=True)
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Blender finished without creating {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_obj"] = str(obj)
    metadata["jaw"] = inspection.get("annotation", {}).get("jaw")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest_path = Path(output_root).resolve() / "dataset_manifest.json"
    manifest = {"samples": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"] = [item for item in manifest.get("samples", []) if item.get("sample_id") != sample_dir.name]
    manifest["samples"].append({"sample_id": sample_dir.name, "path": str(sample_dir), "metadata": "metadata.json"})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return sample_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a five-view Teeth3DS sample")
    parser.add_argument("obj")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--resolution", type=int, default=640)
    args = parser.parse_args()
    print(f"Wrote {build_sample(args.obj, args.output_root, args.blender, args.resolution)}")


if __name__ == "__main__":
    main()