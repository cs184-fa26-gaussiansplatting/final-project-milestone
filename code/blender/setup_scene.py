import json
import math
import os

import bpy

from mathutils import Vector


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        bpy.data.textures.remove(block)
    for block in bpy.data.images:
        bpy.data.images.remove(block)


def setup_hdri(hdri_path=None):
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new(type="ShaderNodeBackground")
    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], output.inputs["Surface"])
    if hdri_path and os.path.exists(hdri_path):
        tex_node = nodes.new(type="ShaderNodeTexEnvironment")
        tex_node.image = bpy.data.images.load(hdri_path)
        links.new(tex_node.outputs["Color"], bg.inputs["Color"])
    else:
        bg.inputs["Color"].default_value = (0.1, 0.1, 0.12, 1.0)
    return world


def import_mesh(mesh_path):
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=mesh_path)
    elif ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    else:
        raise ValueError("Unsupported mesh extension: %s" % ext)
    # return imported mesh objects
    return [o for o in bpy.context.selected_objects if o.type == "MESH"]


def add_camera(location, look_at, name = "Camera"):
    location = Vector(location)
    look_at = Vector(look_at)
    
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = location

    direction = look_at - location
    
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    return cam_obj
    
    # cam_data = bpy.data.cameras.new(name)
    # cam = bpy.data.objects.new(name, cam_data)
    # bpy.context.collection.objects.link(cam)
    # cam.location = location
    # direction = look_at - location
    # rot_quat = direction.to_track_quat("-Z", "Y")
    # cam.rotation_euler = rot_quat.to_euler()
    
    # return cam


def orb_camera_positions(center, radius, n=8):
    positions = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        z = center[2] + radius * 0.35
        positions.append((x, y, z))
    return positions


def setup_render(resolution_x=1024, resolution_y=1024, samples=128):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.cycles.samples = samples
    scene.cycles.device = "GPU" if bpy.context.preferences.addons.get("cycles") else "CPU"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Blender 4.x: import mesh, HDRI, 8-camera orb render.")
    parser.add_argument("--mesh", required=True, help="Path to input mesh (.obj/.ply/.glb)")
    parser.add_argument("--hdri", default=None, help="Optional HDRI environment map path")
    parser.add_argument("--outdir", default="output/renders", help="Render output directory")
    parser.add_argument("--cameras", default="output/cameras.json", help="Camera extrinsic output path")
    parser.add_argument("--radius", type=float, default=3.0, help="Orbit radius")
    parser.add_argument("--num-cams", type=int, default=8, help="Number of orbit cameras")
    parser.add_argument("--resolution", type=int, default=1024, help="Render resolution")
    parser.add_argument("--samples", type=int, default=128, help="Cycles samples")
    # Blender passes everything after '--' to the script
    argv = None
    if "--" in sys_argv():
        argv = sys_argv()[sys_argv().index("--") + 1:]
    args = parser.parse_args(argv)

    clear_scene()
    setup_hdri(args.hdri)
    meshes = import_mesh(args.mesh)

    # compute scene center & radius (pure Python; no numpy dependency in blender)
    center = [0.0, 0.0, 0.0]
    radius = args.radius
    if meshes:
        all_coords = []
        for m in meshes:
            for v in m.data.vertices:
                all_coords.append((v.co.x, v.co.y, v.co.z))
        if all_coords:
            cx = sum(c[0] for c in all_coords) / len(all_coords)
            cy = sum(c[1] for c in all_coords) / len(all_coords)
            cz = sum(c[2] for c in all_coords) / len(all_coords)
            center = [cx, cy, cz]
            max_dist = max(
                math.dist((cx, cy, cz), c) for c in all_coords
            )
            radius = max(args.radius, max_dist * 1.5)

    cams = orb_camera_positions(center, radius, args.num_cams)
    look_at = center

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.cameras)) or ".", exist_ok=True)

    camera_meta = []
    scene = bpy.context.scene
    for i, pos in enumerate(cams):
        cam_name = "Cam%02d" % i
        cam = add_camera(pos, look_at, cam_name)
        scene.camera = cam
        scene.render.filepath = os.path.join(args.outdir, "cam%02d.png" % i)
        bpy.ops.render.render(write_still=True)
        camera_meta.append({
            "index": i,
            "name": cam_name,
            "location": list(pos),
            "look_at": list(look_at),
            "image": "cam%02d.png" % i,
        })

    with open(args.cameras, "w") as f:
        json.dump({"cameras": camera_meta}, f, indent=2)

    print("rendered %d cameras -> %s" % (len(cams), args.outdir))
    print("camera metadata -> %s" % args.cameras)


# small local helper to read argv the way blender passes it (everything after '--')
def sys_argv():
    import sys
    return sys.argv


if __name__ == "__main__":
    main()
