import numpy as np
import trimesh
from pathlib import Path

def create_molar_crown_obj() -> trimesh.Trimesh:
    """Creates a high-precision 3D molar crown model with cusps and grooves."""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=8.0)
    vertices = sphere.vertices.copy()
    
    # Deform sphere into realistic molar crown anatomy
    # Scale width and depth
    vertices[:, 0] *= 1.25  # Mesiodistal width
    vertices[:, 1] *= 1.15  # Buccolingual width
    vertices[:, 2] *= 0.85  # Occlusocervical height
    
    # Sculpt 4 main occlusal cusps (Mesiobuccal, Distobuccal, Mesiolingual, Distolingual)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    top_mask = z > 1.0
    
    # Add cusp elevations
    cusp_mb = np.exp(-((x + 4)**2 + (y - 3)**2) / 12.0) * 2.5
    cusp_db = np.exp(-((x - 4)**2 + (y - 3)**2) / 12.0) * 2.2
    cusp_ml = np.exp(-((x + 4)**2 + (y + 3)**2) / 12.0) * 2.4
    cusp_dl = np.exp(-((x - 4)**2 + (y + 3)**2) / 12.0) * 2.0
    
    # Central fossa depression
    fossa = -np.exp(-(x**2 + y**2) / 16.0) * 1.5
    
    vertices[top_mask, 2] += (cusp_mb + cusp_db + cusp_ml + cusp_dl + fossa)[top_mask]
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=sphere.faces)
    mesh.fix_normals()
    return mesh

def create_incisor_crown_obj() -> trimesh.Trimesh:
    """Creates a 3D anterior incisor crown model with incisal edge."""
    cylinder = trimesh.creation.cylinder(radius=5.0, height=12.0, sections=32)
    vertices = cylinder.vertices.copy()
    
    # Flatten labiolingually toward incisal edge
    z_factor = (vertices[:, 2] + 6.0) / 12.0
    vertices[:, 1] *= (1.0 - 0.6 * z_factor)  # Taper labiolingual width at incisal edge
    vertices[:, 0] *= (1.0 + 0.3 * z_factor)  # Flatten mesiodistal edge
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=cylinder.faces)
    mesh.fix_normals()
    return mesh

def create_premolar_crown_obj() -> trimesh.Trimesh:
    """Creates a bicuspid premolar crown model."""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=6.5)
    vertices = sphere.vertices.copy()
    
    vertices[:, 0] *= 0.95
    vertices[:, 1] *= 1.20
    vertices[:, 2] *= 1.05
    
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    top_mask = z > 0.5
    
    cusp_buccal = np.exp(-((x)**2 + (y - 3.5)**2) / 10.0) * 3.0
    cusp_lingual = np.exp(-((x)**2 + (y + 3.5)**2) / 10.0) * 2.2
    groove = -np.exp(-(x**2 + (y * 0.5)**2) / 8.0) * 1.2
    
    vertices[top_mask, 2] += (cusp_buccal + cusp_lingual + groove)[top_mask]
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=sphere.faces)
    mesh.fix_normals()
    return mesh

def main():
    target_dirs = [Path("data/raw/Teeth3DS"), Path("test_samples")]
    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    models = {
        "molar_crown.obj": create_molar_crown_obj(),
        "premolar_crown.obj": create_premolar_crown_obj(),
        "incisor_crown.obj": create_incisor_crown_obj()
    }
    
    for filename, mesh in models.items():
        for d in target_dirs:
            out_path = d / filename
            mesh.export(str(out_path))
            print(f"Exported test tooth OBJ: {out_path} ({len(mesh.vertices)} vertices)")

if __name__ == "__main__":
    main()
