import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_points(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()

    if isinstance(data, dict):
        if "points" not in data:
            raise ValueError("The .npy file does not contain a 'points' array.")
        points = np.asarray(data["points"], dtype=np.float32)
        colors = data.get("colors")
    else:
        points = np.asarray(data, dtype=np.float32)
        colors = None

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Points must be a 2D array with at least three columns (x, y, z).")

    if colors is None and points.shape[1] >= 6:
        colors = points[:, 3:6]
    points = points[:, :3]

    if colors is None:
        colors = np.full((len(points), 3), 0.5, dtype=np.float32)
    else:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.shape != (len(points), 3):
            raise ValueError("Colors must have shape (N, 3), matching the points.")
        if colors.size and (np.nanmax(colors) > 1.0 or np.nanmin(colors) < 0.0):
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

    finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    return points[finite], colors[finite]


def main():
    project_root = Path(__file__).resolve().parents[2]
    default_points = project_root / "output" / "points.npy"

    parser = argparse.ArgumentParser(description="Interactively view a sampled point cloud.")
    parser.add_argument(
        "points_npy",
        nargs="?",
        default=str(default_points),
        help="Input .npy file (default: output/points.npy)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=100000,
        help="Maximum number of points to display (default: 100000)",
    )
    args = parser.parse_args()
    if args.max_points < 1:
        parser.error("--max-points must be at least 1")

    points_path = Path(args.points_npy).expanduser()
    if not points_path.exists():
        parser.error("Point cloud file not found: %s" % points_path)

    points, colors = load_points(points_path)
    if len(points) == 0:
        raise RuntimeError("The point cloud contains no finite points.")

    if len(points) > args.max_points:
        step = (len(points) + args.max_points - 1) // args.max_points
        points = points[::step]
        colors = colors[::step]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=2,
        linewidths=0,
        depthshade=True,
    )

    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    span = np.max(points.max(axis=0) - points.min(axis=0))
    if span > 0:
        ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point cloud: %d points" % len(points))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()