import argparse
import open3d as o3d
from pathlib import Path

def convert_ply_to_obj(sample_dir: str | Path):
    sample = Path(sample_dir)
    ply_path = sample / "registered_reconstruction.ply"
    obj_path = sample / "reconstruction_surface.obj"
    
    if not ply_path.exists():
        print(f"Error: {ply_path} not found.")
        return

    print(f"Loading point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    # Estimate normals if they don't exist
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)
        
    print("Running Poisson surface reconstruction to create a solid mesh...")
    # depth=9 is a good balance between detail and performance
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # Optional: Crop the mesh to remove low-density artifact areas (the "bubble" effect)
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    print(f"Saving solid OBJ mesh to {obj_path}...")
    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    print("Done! You can now view this .obj file.")

def main():
    parser = argparse.ArgumentParser(description="Convert PLY point cloud to a solid OBJ mesh")
    parser.add_argument("sample")
    args = parser.parse_args()
    convert_ply_to_obj(args.sample)

if __name__ == "__main__":
    main()
