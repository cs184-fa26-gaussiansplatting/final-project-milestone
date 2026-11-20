import argparse
import os
import struct
import time

import numpy as np


def write_ply(path, positions, colors, normals=None, scale=0.005, opacity=0.95):
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

    # default scale per point
    scales = np.full((n, 3), scale, dtype=np.float32)
    # quaternion from normal: rotate z-axis to normal
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 3] = 1.0  # w
    z = np.array([0, 0, 1], dtype=np.float32)
    for i in range(n):
        nrm = normals[i]
        v = np.cross(z, nrm)
        s = np.dot(z, nrm)
        if s < -0.99999:
            q = np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            q = np.array([v[0], v[1], v[2], 1 + s], dtype=np.float32)
            q = q / (np.linalg.norm(q) + 1e-8)
        quats[i] = q

    opacities = np.full((n, 1), opacity, dtype=np.float32)
    # SH DC: color = 0.5 + C0 * sh_dc  =>  sh_dc = (color - 0.5) / C0
    C0 = 0.28209479177387814
    sh_dc = (colors - 0.5) / C0

    # PLY header (binary little endian, matches gsplat / SuperSplat layout)
    header = """ply
format binary_little_endian 1.0
element vertex %d
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
""" % n

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            row = struct.pack(
                "<fffffffffffffffff",
                positions[i, 0], positions[i, 1], positions[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2],
                opacities[i, 0],
                scales[i, 0], scales[i, 1], scales[i, 2],
                quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3],
            )
            f.write(row)


def main():
    parser = argparse.ArgumentParser(
        description="Convert sampled points to a Gaussian-splat .ply (binary, gsplat layout)."
    )
    parser.add_argument("points_npy", help="Input points.npy from sample.py")
    parser.add_argument(
        "-o", "--out", default="output/splat.ply", help="Output .ply path"
    )
    parser.add_argument(
        "--scale", type=float, default=0.005, help="Per-point scale (default 0.005)"
    )
    parser.add_argument(
        "--opacity", type=float, default=0.95, help="Per-point opacity (default 0.95)"
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    data = np.load(args.points_npy, allow_pickle=True).item()
    points = data["points"]
    colors = data["colors"]
    normals = data.get("normals", None)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_ply(args.out, points, colors, normals, scale=args.scale, opacity=args.opacity)
    total = time.perf_counter() - t0

    print("converted %d points -> %s (scale=%.4f opacity=%.2f) in %.3fs"
          % (points.shape[0], args.out, args.scale, args.opacity, total))


if __name__ == "__main__":
    main()
