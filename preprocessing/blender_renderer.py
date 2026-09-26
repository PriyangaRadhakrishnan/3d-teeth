"""Blender 4 renderer for RGB, metric depth, camera-space normals, and masks."""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import bpy
from mathutils import Vector

DENTAL_VIEWS = {
    "frontal": (0, 0, 1),
    "left_buccal": (-1, 0.2, 0.4),
    "right_buccal": (1, 0.2, 0.4),
    "maxillary_occlusal": (0, -1, 0.1),
    "mandibular_occlusal": (0, 1, 0.1)
}

LEGACY_VIEWS = {
    "top": (0, -1, 0),
    "front": (0, 0, 1),
    "back": (0, 0, -1),
    "left": (-1, 0, 0),
    "right": (1, 0, 0)
}

def make_material(name, color, camera_normal=False, specular=0.5):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    
    if camera_normal:
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0
        geometry = nodes.new("ShaderNodeNewGeometry")
        transform = nodes.new("ShaderNodeVectorTransform")
        transform.vector_type = "NORMAL"
        transform.convert_from = "WORLD"
        transform.convert_to = "CAMERA"
        scale = nodes.new("ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.inputs[3].default_value = 0.5
        offset = nodes.new("ShaderNodeVectorMath")
        offset.operation = "ADD"
        offset.inputs[1].default_value = (0.5, 0.5, 0.5)
        links.new(geometry.outputs["Normal"], transform.inputs[0])
        links.new(transform.outputs[0], scale.inputs[0])
        links.new(scale.outputs[0], offset.inputs[0])
        links.new(offset.outputs[0], emission.inputs["Color"])
        links.new(emission.outputs[0], output.inputs[0])
    else:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.25
        bsdf.inputs["Specular IOR Level"].default_value = specular
        links.new(bsdf.outputs[0], output.inputs[0])
        
    return material

def setup_lights(scene, center, extent):
    # Studio 3-Point Dental Lighting
    key_data = bpy.data.lights.new("KeyLight", "SUN")
    key_data.energy = 4.5
    key_light = bpy.data.objects.new("KeyLight", key_data)
    key_light.location = center + Vector((extent, -extent, extent * 1.5))
    key_light.rotation_euler = (0.6, 0.4, 0.0)
    scene.collection.objects.link(key_light)

    fill_data = bpy.data.lights.new("FillLight", "SUN")
    fill_data.energy = 2.2
    fill_light = bpy.data.objects.new("FillLight", fill_data)
    fill_light.location = center + Vector((-extent, -extent, extent))
    fill_light.rotation_euler = (0.5, -0.5, 0.0)
    scene.collection.objects.link(fill_light)

def render_scene(mesh_path, output_dir, resolution):
    output = Path(output_dir)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.world.color = (0.02, 0.02, 0.03)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    try:
        bpy.ops.wm.obj_import(filepath=str(mesh_path))
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=str(mesh_path))

    mesh_objects = [item for item in scene.objects if item.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("OBJ produced no mesh object")

    mesh = mesh_objects[0]
    bounds = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
    center = sum(bounds, Vector()) / 8.0
    extent = max(max(value[i] for value in bounds) - min(value[i] for value in bounds) for i in range(3))

    setup_lights(scene, center, extent)

    camera_data = bpy.data.cameras.new("GroundTruthCamera")
    camera_data.lens = 52
    camera_data.sensor_width = 36
    camera = bpy.data.objects.new("GroundTruthCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    rgb = make_material("DentalEnamel", (0.88, 0.90, 0.94), specular=0.7)
    mask = make_material("Mask", (1.0, 1.0, 1.0))
    normal = make_material("CameraNormal", (0.5, 0.5, 1.0), camera_normal=True)

    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    compositor = scene.node_tree
    compositor.nodes.clear()

    render_layers = compositor.nodes.new("CompositorNodeRLayers")
    depth_output = compositor.nodes.new("CompositorNodeOutputFile")
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_mode = "BW"
    depth_output.base_path = str(output / "depth")
    compositor.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])

    depth_cap = compositor.nodes.new("CompositorNodeMath")
    depth_cap.operation = "MINIMUM"
    depth_cap.inputs[1].default_value = extent * 3.0

    depth_range = compositor.nodes.new("CompositorNodeMapRange")
    depth_range.inputs["From Min"].default_value = 0.0
    depth_range.inputs["From Max"].default_value = extent * 3.0
    depth_range.inputs["To Min"].default_value = 1.0
    depth_range.inputs["To Max"].default_value = 0.0

    depth_preview = compositor.nodes.new("CompositorNodeOutputFile")
    depth_preview.format.file_format = "PNG"
    depth_preview.base_path = str(output / "depth_vis")
    compositor.links.new(render_layers.outputs["Depth"], depth_cap.inputs[0])
    compositor.links.new(depth_cap.outputs[0], depth_range.inputs["Value"])
    compositor.links.new(depth_range.outputs["Value"], depth_preview.inputs[0])

    cameras = {}
    
    # Render all dental & legacy views
    all_views = {**DENTAL_VIEWS, **LEGACY_VIEWS}
    
    for view_name, direction_tuple in all_views.items():
        direction = Vector(direction_tuple).normalized()
        camera.location = center + direction * (extent * 1.45)
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()

        mesh.data.materials.clear()
        mesh.data.materials.append(rgb)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(output / "images" / f"{view_name}.png")
        bpy.ops.render.render(write_still=True)

        mesh.data.materials.clear()
        mesh.data.materials.append(mask)
        scene.render.filepath = str(output / "masks" / f"{view_name}.png")
        bpy.ops.render.render(write_still=True)

        mesh.data.materials.clear()
        mesh.data.materials.append(normal)
        scene.render.filepath = str(output / "normals" / f"{view_name}.png")
        bpy.ops.render.render(write_still=True)

        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.filepath = str(output / "depth" / f"_{view_name}_render.png")
        depth_output.file_slots[0].path = view_name
        depth_preview.file_slots[0].path = view_name
        bpy.ops.render.render(write_still=True)

        generated_depth = next((item for item in (output / "depth").glob(f"{view_name}*.exr") if item.name != f"{view_name}.exr"), None)
        if generated_depth:
            shutil.move(str(generated_depth), str(output / "depth" / f"{view_name}.exr"))

        for extra in (output / "depth").glob("*.exr"):
            if extra.name != f"{view_name}.exr" and extra.name.startswith(view_name):
                extra.unlink()

        scene.render.image_settings.file_format = "PNG"
        focal = (resolution / 2.0) / math.tan(math.radians(39.0) / 2.0)
        cameras[view_name] = {
            "position": list(camera.location),
            "rotation_euler": list(camera.rotation_euler),
            "intrinsics": [[focal, 0, resolution / 2], [0, focal, resolution / 2], [0, 0, 1]],
            "focal_length_pixels": focal,
            "principal_point": [resolution / 2, resolution / 2],
            "field_of_view_degrees": 39.0,
            "extrinsics_world_to_camera": [list(row) for row in camera.matrix_world.inverted()],
            "image_width": resolution,
            "image_height": resolution
        }

    (output / "cameras" / "cameras.json").write_text(json.dumps(cameras, indent=2), encoding="utf-8")
    metadata = {
        "sample_id": output.name,
        "views": list(DENTAL_VIEWS.keys()),
        "resolution": [resolution, resolution],
        "camera_file": "cameras/cameras.json",
        "image_files": [f"images/{v}.png" for v in DENTAL_VIEWS],
        "depth_files": [f"depth/{v}.exr" for v in DENTAL_VIEWS],
        "depth_visualization_files": [f"depth_vis/{v}.png" for v in DENTAL_VIEWS],
        "normal_files": [f"normals/{v}.png" for v in DENTAL_VIEWS],
        "mask_files": [f"masks/{v}.png" for v in DENTAL_VIEWS],
        "normal_encoding": "RGB = camera-space XYZ * 0.5 + 0.5",
        "depth_encoding": "metric camera Z depth in EXR"
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=640)
    blender_separator = sys.argv.index("--") if "--" in sys.argv else 0
    args = parser.parse_args(sys.argv[blender_separator + 1:])
    render_scene(args.mesh, args.output, args.resolution)

if __name__ == "__main__":
    main()