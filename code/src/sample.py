import argparse
import os
import struct
import sys
import time

import numpy as np


def parse_mtl(path):
    """Parse a .mtl file into a dict of material name -> diffuse color (rgb 0..1)."""
    materials = {}
    current = None
    if not path or not os.path.exists(path):
        return materials
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "newmtl":
                current = parts[1]
                materials[current] = {"Kd": np.array([0.7, 0.7, 0.7], dtype=np.float32)}
            elif parts[0] == "Kd" and current is not None:
                materials[current]["Kd"] = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32
                )
    return materials


def load_obj(obj_path):
    """Load a .obj (with sidecar .mtl) into vertices, faces, per-vertex normals, per-face material colors."""
    mtl_path = obj_path.rsplit(".", 1)[0] + ".mtl"
    materials = parse_mtl(mtl_path)

    positions = []
    normals = []
    texcoords = []
    faces = []  # list of (pos_idx_list, normal_idx_list, tex_idx_list, material_name)

    current_material = "default"
    materials.setdefault(
        "default", {"Kd": np.array([0.7, 0.7, 0.7], dtype=np.float32)}
    )

    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            prefix = parts[0]
            if prefix == "v":
                positions.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            elif prefix == "vn":
                normals.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            elif prefix == "vt":
                texcoords.append(
                    [float(parts[1]), float(parts[2])]
                    + ([float(parts[3])] if len(parts) > 3 else [0.0])
                )
            elif prefix == "usemtl":
                current_material = parts[1]
                if current_material not in materials:
                    materials[current_material] = {
                        "Kd": np.array([0.7, 0.7, 0.7], dtype=np.float32)
                    }
            elif prefix == "f":
                # face vertices may be v, v/vt, v//vn, or v/vt/vn
                verts = parts[1:]
                pos_idxs = []
                nor_idxs = []
                tex_idxs = []
                for v in verts:
                    comps = v.split("/")
                    pos_idxs.append(int(comps[0]) - 1 if comps[0] else -1)
                    tex_idxs.append(int(comps[1]) - 1 if len(comps) > 1 and comps[1] else -1)
                    nor_idxs.append(int(comps[2]) - 1 if len(comps) > 2 and comps[2] else -1)
                # triangulate fan
                for i in range(1, len(verts) - 1):
                    tri = [
                        (pos_idxs[0], nor_idxs[0], tex_idxs[0]),
                        (pos_idxs[i], nor_idxs[i], tex_idxs[i]),
                        (pos_idxs[i + 1], nor_idxs[i + 1], tex_idxs[i + 1]),
                    ]
                    faces.append((tri, current_material))

    positions = np.asarray(positions, dtype=np.float32) if positions else np.zeros((0, 3), dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32) if normals else None
    return positions, normals, faces, materials


def triangle_area(p0, p1, p2):
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))


def face_normal(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return n / norm


def sample_points(positions, normals, faces, materials, n_points, seed=0):
    rng = np.random.default_rng(seed)

    # Build per-face data: positions, area, color, normal
    face_positions = []
    face_areas = []
    face_colors = []
    face_normals = []
    degenerate = 0

    for tri, mat_name in faces:
        (p0i, n0i, _), (p1i, n1i, _), (p2i, n2i, _) = tri
        if p0i < 0 or p1i < 0 or p2i < 0:
            degenerate += 1
            continue
        p0 = positions[p0i]
        p1 = positions[p1i]
        p2 = positions[p2i]
        area = triangle_area(p0, p1, p2)
        if area < 1e-12:
            degenerate += 1
            continue
        face_positions.append((p0, p1, p2))
        face_areas.append(area)
        kd = materials.get(mat_name, materials["default"])["Kd"]
        face_colors.append(kd)
        # per-vertex normal averaging fallback: face normal
        face_normals.append(face_normal(p0, p1, p2))

    if len(face_areas) == 0:
        raise RuntimeError("No valid (non-degenerate) triangles found in mesh.")

    # stack face data into arrays: (F,3,3) positions, (F,3) colors/normals
    face_pos_arr = np.asarray(face_positions, dtype=np.float32)
    face_areas = np.asarray(face_areas, dtype=np.float64)
    face_colors = np.asarray(face_colors, dtype=np.float32)
    face_normals = np.asarray(face_normals, dtype=np.float32)

    # Area-weighted sampling: pick triangle with probability proportional to area
    probs = face_areas / face_areas.sum()
    tri_choice = rng.choice(len(face_areas), size=n_points, p=probs)
    # Uniform barycentric sampling within each chosen triangle
    u = rng.random((n_points, 2))
    sqrt_u1 = np.sqrt(u[:, 0])
    a = 1.0 - sqrt_u1
    b = sqrt_u1 * (1.0 - u[:, 1])
    c = sqrt_u1 * u[:, 1]

    p0 = face_pos_arr[tri_choice, 0]  # (N,3)
    p1 = face_pos_arr[tri_choice, 1]
    p2 = face_pos_arr[tri_choice, 2]

    points = a[:, None] * p0 + b[:, None] * p1 + c[:, None] * p2
    colors = face_colors[tri_choice]
    normals_out = face_normals[tri_choice]

    return points, colors, normals_out, degenerate


def write_ply(path, positions, colors, normals=None):
    n = positions.shape[0]
    positions = np.asarray(positions, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)
    if normals is None:
        normals = np.zeros_like(positions)
    else:
        normals = np.asarray(normals, dtype=np.float32)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

    header = (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex %d\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float red\n"
        "property float green\n"
        "property float blue\n"
        "end_header\n"
    ) % n

    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(
                "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"
                % (
                    positions[i, 0],
                    positions[i, 1],
                    positions[i, 2],
                    normals[i, 0],
                    normals[i, 1],
                    normals[i, 2],
                    colors[i, 0],
                    colors[i, 1],
                    colors[i, 2],
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description="Area-weighted barycentric sampling of an .obj mesh to a point cloud."
    )
    parser.add_argument("obj", help="Input .obj mesh file")
    parser.add_argument(
        "-o", "--out", default="output/points.npy", help="Output .npy path"
    )
    parser.add_argument(
        "--ply", default=None, help="Optional .ply preview output path"
    )
    parser.add_argument(
        "-n", "--num-points", type=int, default=10000, help="Number of points to sample"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed"
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    positions, normals, faces, materials = load_obj(args.obj)
    t_load = time.perf_counter() - t0

    print("loaded mesh: %d vertices, %d faces" % (positions.shape[0], len(faces)))

    t1 = time.perf_counter()
    points, colors, normals_out, degenerate = sample_points(
        positions, normals, faces, materials, args.num_points, seed=args.seed
    )
    t_sample = time.perf_counter() - t1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    t_save0 = time.perf_counter()
    np.save(args.out, {"points": points, "colors": colors, "normals": normals_out})
    t_save = time.perf_counter() - t_save0

    if args.ply:
        write_ply(args.ply, points, colors, normals_out)

    total = time.perf_counter() - t0
    print("sampled %d points (skipped %d degenerate faces)" % (args.num_points, degenerate))
    print("timing: load=%.3fs sample=%.3fs save=%.3fs total=%.3fs"
          % (t_load, t_sample, t_save, total))


if __name__ == "__main__":
    main()
