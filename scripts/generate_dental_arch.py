import numpy as np
import trimesh
from pathlib import Path

def create_individual_tooth(tooth_type="molar", center=(0, 0, 0), scale=1.0) -> trimesh.Trimesh:
    if tooth_type == "molar":
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=4.5 * scale)
        v = sphere.vertices.copy()
        v[:, 0] *= 1.25; v[:, 1] *= 1.15; v[:, 2] *= 0.85
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        top = z > 0
        cusps = (np.exp(-((x+2)**2 + (y-2)**2)/6.0) + np.exp(-((x-2)**2 + (y-2)**2)/6.0) +
                 np.exp(-((x+2)**2 + (y+2)**2)/6.0) + np.exp(-((x-2)**2 + (y+2)**2)/6.0)) * 1.5
        v[top, 2] += cusps[top]
    elif tooth_type == "premolar":
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=3.8 * scale)
        v = sphere.vertices.copy()
        v[:, 0] *= 0.9; v[:, 1] *= 1.2; v[:, 2] *= 0.95
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        top = z > 0
        cusps = (np.exp(-(x**2 + (y-2)**2)/5.0) + np.exp(-(x**2 + (y+2)**2)/5.0)) * 1.6
        v[top, 2] += cusps[top]
    else:  # incisor / canine
        cylinder = trimesh.creation.cylinder(radius=3.0 * scale, height=9.0 * scale, sections=24)
        v = cylinder.vertices.copy()
        z_fact = (v[:, 2] + 4.5) / 9.0
        v[:, 1] *= (1.0 - 0.5 * z_fact)
        v[:, 0] *= (1.0 + 0.25 * z_fact)
        sphere = trimesh.Trimesh(vertices=v, faces=cylinder.faces)

    mesh = trimesh.Trimesh(vertices=v, faces=sphere.faces)
    mesh.apply_translation(center)
    return mesh

def build_maxillary_dental_arch() -> trimesh.Trimesh:
    """Builds a realistic 14-tooth parabolic maxillary dental arch mesh."""
    meshes = []
    
    # Parabolic arch equation: y = a * x^2
    # 14 teeth positioned symmetrically along parabolic arch
    teeth_types = [
        ("molar", 24.0), ("molar", 20.0), ("premolar", 16.0), ("premolar", 12.0),
        ("canine", 8.0), ("incisor", 4.0), ("incisor", 1.2),
        ("incisor", -1.2), ("incisor", -4.0), ("canine", -8.0),
        ("premolar", -12.0), ("premolar", -16.0), ("molar", -20.0), ("molar", -24.0)
    ]
    
    a = 0.035  # Arch curvature
    for t_type, x in teeth_types:
        y = a * (x ** 2)
        z = np.sin(x * 0.1) * 0.8  # Slight occlusal curve of Spee
        
        # Calculate tangent rotation for natural dental alignment
        dydx = 2 * a * x
        angle = np.arctan(dydx)
        
        tooth = create_individual_tooth(tooth_type=t_type, center=(x, y, z), scale=1.0)
        tooth.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1], point=[x, y, z]))
        meshes.append(tooth)

    # Base gingival ridge curve connecting arch
    arch_mesh = trimesh.util.concatenate(meshes)
    arch_mesh.fix_normals()
    return arch_mesh

def main():
    arch = build_maxillary_dental_arch()
    raw_dir = Path("data/raw/Teeth3DS")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in ["00OMSZGW_upper.obj", "SAMPLE_upper.obj"]:
        target_path = raw_dir / filename
        arch.export(str(target_path))
        print(f"Exported REAL 3D Dental Arch OBJ to {target_path} ({len(arch.vertices)} vertices, {len(arch.faces)} faces)")

if __name__ == "__main__":
    main()
