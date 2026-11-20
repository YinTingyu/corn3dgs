#!/usr/bin/env python3
"""Check the cameras of a COLMAP model: intrinsics, cameras that sit far off the orbit, and sudden jumps
between frames. Also saves a plot of the camera path.

    python scripts/tools/check_cameras.py <clip>/colmap_bruteforce_out/sparse/0  [more models to compare ...]

Each model is first converted to text with `colmap model_converter` into a temp folder (the original is not
touched). One bad camera far away from the others is enough to break nerfstudio's automatic scaling, and the
splat then comes out tiny: that is what happened to 14h 2026-09-04 (frame_00489 at 29.6x the median distance).
The fix is `colmap image_deleter` on that frame, then retrain.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def to_txt(model_dir):
    if os.path.exists(os.path.join(model_dir, "images.txt")):
        return model_dir
    out = tempfile.mkdtemp(prefix="colmap_txt_")
    subprocess.run(["colmap", "model_converter", "--input_path", model_dir, "--output_path", out,
                    "--output_type", "TXT"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def read_model(txt_dir):
    cams = [line.split() for line in open(os.path.join(txt_dir, "cameras.txt"))
            if not line.startswith("#") and line.strip()]
    names, centers, views = [], [], []
    lines = [l for l in open(os.path.join(txt_dir, "images.txt")) if not l.startswith("#")]
    for i in range(0, len(lines), 2):                # two lines per image, the first one is the pose
        f = lines[i].split()
        if len(f) < 10:
            continue
        q = np.array(list(map(float, f[1:5])))
        t = np.array(list(map(float, f[5:8])))
        R = qvec2rotmat(q)
        centers.append(-R.T @ t)                     # camera position in world coordinates
        views.append(R.T @ np.array([0, 0, 1.0]))    # viewing direction
        names.append(f[9])
    order = np.argsort(names)
    return cams, [names[k] for k in order], np.array(centers)[order], np.array(views)[order]


def check(model):
    tag = model.rstrip("/").replace("/colmap_bruteforce_out/sparse/0", "").split("/")
    tag = "_".join(tag[-2:]) if len(tag) >= 2 else tag[-1]
    cams, names, C, V = read_model(to_txt(model))
    print(f"\n########## {tag}")
    for c in cams:
        print(f"  camera {c[0]}: {c[1]}  {c[2]}x{c[3]}  params = {', '.join(f'{float(p):.4g}' for p in c[4:])}")
    print(f"  registered frames: {len(C)}")

    # orbit center = the point closest to all viewing rays (least squares)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, v in zip(C, V):
        Pm = np.eye(3) - np.outer(v, v)
        A += Pm
        b += Pm @ c
    center = np.linalg.solve(A, b)
    r = np.linalg.norm(C - center, axis=1)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826 + 1e-12
    out = np.flatnonzero(np.abs(r - med) > 5 * mad + 0.5 * med)
    print(f"  distance camera -> orbit center: median {med:.3f}, 5-95% {np.percentile(r, 5):.3f}-"
          f"{np.percentile(r, 95):.3f}, max {r.max():.3f} ({r.max() / med:.1f}x median)")
    if len(out):
        print(f"  OUTLIER cameras ({len(out)}): " + ", ".join(f"{names[k]} ({r[k] / med:.1f}x)" for k in out[:15])
              + (" ..." if len(out) > 15 else ""))
    else:
        print("  no outlier cameras")
    # a pose that jumps far from the previous frame means that part was registered wrongly
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    big = np.flatnonzero(step > 10 * np.median(step))
    print(f"  frame-to-frame jump: median {np.median(step):.4f}; jumps >10x median: {len(big)}"
          + (" at " + ", ".join(f"{names[k]}->{names[k + 1]}" for k in big[:10]) if len(big) else ""))

    # plot the path on the main axes of the camera positions
    Y = (C - center) @ np.linalg.svd(C - center, full_matrices=False)[2].T
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for a, (i, j, nm) in zip(axes, [(0, 1, "orbit plane"), (0, 2, "side")]):
        sc = a.scatter(Y[:, i], Y[:, j], c=np.arange(len(Y)), s=6, cmap="viridis")
        if len(out):
            a.scatter(Y[out, i], Y[out, j], s=40, facecolors="none", edgecolors="red")
        a.scatter([0], [0], marker="+", c="k", s=80)
        a.set_aspect("equal")
        a.set_title(f"{tag}: {nm} (color = frame order)")
    plt.colorbar(sc, ax=axes, shrink=0.8, label="frame index")
    png = f"cameras_{tag}.png"
    plt.savefig(png, dpi=130)
    plt.close(fig)
    print(f"  saved {png}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for m in sys.argv[1:]:
        check(m)
