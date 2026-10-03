"""Writes a synthetic capture bundle with known geometry: a sloped driveway, a barrel and a wall,
plus keyframes rendered by ray casting. Used by test_pipeline.py.

Usage: python tests/make_fake_bundle.py OUT_DIR
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

import pillow_heif

pillow_heif.register_heif_opener()

SLOPE = 0.03
BARREL = np.array([2.0, 3.0])
BARREL_R, BARREL_H = 0.3, 0.9
IMG_W, IMG_H, DEPTH_W, DEPTH_H = 192, 144, 64, 48
K = np.array([[150.0, 0, 96], [0, 150.0, 72], [0, 0, 1]])


def ground_z(x, y):
    return SLOPE * np.asarray(y) + 0.01 * np.sin(np.asarray(x))


def write_glb(path: Path, parts: list[tuple[str, np.ndarray, np.ndarray]]):
    binary, views, accessors, meshes, nodes = b"", [], [], [], []
    for name, v, f in parts:
        pos = v.astype("<f4").tobytes()
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(pos)})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5126, "count": len(v), "type": "VEC3",
                          "min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist()})
        binary += pos
        idx = f.astype("<u4").tobytes()
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(idx)})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5125, "count": f.size, "type": "SCALAR"})
        binary += idx
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": len(accessors) - 2}, "indices": len(accessors) - 1}]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})
    doc = json.dumps({"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
                      "nodes": nodes, "meshes": meshes, "accessors": accessors, "bufferViews": views,
                      "buffers": [{"byteLength": len(binary)}]}).encode()
    doc += b" " * (-len(doc) % 4)
    binary += b"\0" * (-len(binary) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(doc) + 8 + len(binary))
    out += struct.pack("<II", len(doc), 0x4E4F534A) + doc + struct.pack("<II", len(binary), 0x004E4942) + binary
    path.write_bytes(out)


def grid_ground(cell=0.1, x=(-4, 8), y=(-2, 8)):
    xs, ys = np.arange(x[0], x[1] + 1e-9, cell), np.arange(y[0], y[1] + 1e-9, cell)
    gx, gy = np.meshgrid(xs, ys)
    rng = np.random.default_rng(0)
    v = np.column_stack([gx.ravel(), gy.ravel(), ground_z(gx, gy).ravel() + rng.normal(0, 0.003, gx.size)])
    nx = len(xs)
    i = (np.arange(len(ys) - 1)[:, None] * nx + np.arange(nx - 1)[None, :]).ravel()
    f = np.concatenate([np.column_stack([i, i + 1, i + nx + 1]), np.column_stack([i, i + nx + 1, i + nx])])
    return v, f


def look_at(eye, target):
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, up, -fwd, eye  # OpenGL camera: looks along -z
    return T


DRIFT_SHIFT = np.array([0.25, 0.10, 0.0])
DRIFT_YAW_DEG = 1.5
START = np.array([0.0, -1.0, 0.0])


def drift_transform() -> np.ndarray:
    """Pose error on the return trip: yaw about the start point plus a shift."""
    a = np.radians(DRIFT_YAW_DEG)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    D = np.eye(4)
    D[:3, :3] = R
    D[:3, 3] = START - R @ START + DRIFT_SHIFT
    return D


def main(out: Path, path_mode: bool = False, turnaround: bool = True):
    root = out / (("fake-path.scan" if turnaround else "fake-path-noturn.scan") if path_mode else "fake-driveway.scan")
    (root / "keyframes").mkdir(parents=True, exist_ok=True)
    (root / "objects").mkdir(exist_ok=True)

    gv, gf = grid_ground()
    base_z = float(ground_z(*BARREL))
    barrel = trimesh.creation.cylinder(radius=BARREL_R, height=BARREL_H, sections=48).subdivide().subdivide()
    barrel.apply_translation([BARREL[0], BARREL[1], base_z + BARREL_H / 2])
    keep = barrel.face_normals[:, 2] > -0.9  # the phone never sees the bottom
    bv, bf = barrel.vertices, barrel.faces[keep]
    wall = trimesh.creation.box(extents=[6, 0.2, 2]).subdivide().subdivide().subdivide()
    wall.apply_translation([2, 7.5, ground_z(2, 7.5) + 1])
    # (lo, hi) boxes for the renderer; the same boxes go in the mesh.
    boxes = [(np.array([-1.0, 7.4, ground_z(2, 7.5)]), np.array([5.0, 7.6, ground_z(2, 7.5) + 2.0])),   # wall
             (np.array([-1.7, -3.0, -0.2]), np.array([-1.5, 8.0, 0.2])),                               # curb
             (np.array([-1.15, 1.35, -0.1]), np.array([-0.85, 1.65, 1.0])),                            # post
             (np.array([0.65, -3.15, -0.2]), np.array([0.95, -2.85, 1.0]))]                            # post behind start
    extra = []
    for lo, hi in boxes[1:]:
        b = trimesh.creation.box(bounds=[lo, hi]).subdivide().subdivide().subdivide()
        extra.append(b)

    # Tag sphere (r = 0.6) also grabbed the ground around the barrel; that ground is not in mesh.glb.
    centroids = gv[gf].mean(axis=1)
    near = np.linalg.norm(centroids[:, :2] - BARREL, axis=1) < 0.6
    under = np.linalg.norm(centroids[:, :2] - BARREL, axis=1) < BARREL_R  # unseen under the barrel
    write_glb(root / "mesh.glb", [("none", gv, gf[~near]), ("wall", wall.vertices, wall.faces),
                                  *[(f"box{i}", b.vertices, b.faces) for i, b in enumerate(extra)]])
    obj_v = np.concatenate([bv, gv])
    obj_f = np.concatenate([bf, gf[near & ~under] + len(bv)])
    write_glb(root / "objects" / "barrel.glb", [("barrel", obj_v, obj_f)])

    # Render keyframes by exact ray casting against the analytic scene (ground surface, barrel, wall).

    def render(T, w, h, k):
        uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        d_cam = np.stack([(uu - k[0, 2]) / k[0, 0], -(vv - k[1, 2]) / k[1, 1], -np.ones_like(uu)], -1).reshape(-1, 3)
        d = d_cam @ T[:3, :3].T  # unnormalized: t is then the depth along the view axis
        o = T[:3, 3]
        n = len(d)
        t_hit = np.full(n, np.inf)
        col = np.zeros((n, 3), dtype=np.uint8)

        # Ground: march then interpolate the crossing.
        ts = np.arange(0.05, 25, 0.05)
        for a in range(0, n, 4096):
            dd = d[a:a + 4096]
            p = o + ts[None, :, None] * dd[:, None, :]
            above = p[..., 2] - ground_z(p[..., 0], p[..., 1])
            below = above <= 0
            hit = below.any(axis=1)
            i = np.argmax(below, axis=1)
            rows = np.nonzero(hit & (i > 0))[0]
            i = i[rows]
            a0, a1 = above[rows, i - 1], above[rows, i]
            t = ts[i - 1] + 0.05 * a0 / (a0 - a1)
            t_hit[a + rows] = t
            q = o + t[:, None] * dd[rows]
            checker = ((np.floor(q[:, 0]) + np.floor(q[:, 1])) % 2).astype(bool)
            col[a + rows] = np.where(checker[:, None], [90, 160, 80], [150, 150, 140])

        # Barrel side and top cap.
        zb, zt = base_z, base_z + BARREL_H
        ox, oy = o[0] - BARREL[0], o[1] - BARREL[1]
        A = d[:, 0] ** 2 + d[:, 1] ** 2
        B = 2 * (ox * d[:, 0] + oy * d[:, 1])
        C = ox ** 2 + oy ** 2 - BARREL_R ** 2
        disc = B ** 2 - 4 * A * C
        with np.errstate(invalid="ignore", divide="ignore"):
            t_side = (-B - np.sqrt(disc)) / (2 * A)
            z_side = o[2] + t_side * d[:, 2]
            side = (disc > 0) & (t_side > 0) & (z_side >= zb) & (z_side <= zt)
            t_top = (zt - o[2]) / d[:, 2]
            q = o + t_top[:, None] * d
            top = (t_top > 0) & ((q[:, 0] - BARREL[0]) ** 2 + (q[:, 1] - BARREL[1]) ** 2 <= BARREL_R ** 2)
        t_b = np.where(side, t_side, np.inf)
        t_b = np.where(top & (t_top < t_b), t_top, t_b)

        # Boxes (wall, curb, post): slab test.
        hits = [(t_b, [40, 80, 220])]
        for (lo, hi), rgb in zip(boxes, [[200, 60, 50], [230, 230, 220], [120, 70, 30], [120, 70, 30]]):
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / d, (hi - o) / d
            t_near = np.nanmax(np.minimum(t1, t2), axis=1)
            t_far = np.nanmin(np.maximum(t1, t2), axis=1)
            hits.append((np.where((t_near <= t_far) & (t_near > 0), t_near, np.inf), rgb))

        for t_obj, rgb in hits:
            closer = t_obj < t_hit
            t_hit[closer] = t_obj[closer]
            col[closer] = rgb
        depth = np.where(np.isfinite(t_hit), t_hit, 0).astype(np.float32)
        return col.reshape(h, w, 3), depth.reshape(h, w)

    kd = K.copy()
    kd[:2] *= DEPTH_W / IMG_W
    target = np.array([BARREL[0], BARREL[1], base_z + 0.45])
    entries = []
    if path_mode:
        poses = []
        e0 = np.array([0.0, -1.0, 1.3])
        for look in [[0, -2.5, -1.3], [-2.0, -1.5, -1.3], [2.0, -1.5, -1.3]]:  # look around at the start first
            poses.append(look_at(e0, e0 + look))
        for y in np.arange(-1.0, 7.01, 0.5):                       # out, looking ahead
            e = np.array([0.0, y, 1.3])
            poses.append(look_at(e, e + [0, 2.5, -1.3]))
        for y in np.arange(7.0, -1.01, -0.5):                      # back, looking ahead
            e = np.array([0.0, y, 1.3])
            poses.append(look_at(e, e + [0, -2.5, -1.3]))
        n_out = 3 + len(np.arange(-1.0, 7.01, 0.5))
        for y in ([-1.0, -0.5, 0.0, 0.5] if turnaround else []):    # turn around at the start and look again
            e = np.array([0.0, y, 1.3])
            poses.append(look_at(e, e + [0, 2.5, -1.3]))
        D = drift_transform()
        recorded = [T if i < n_out else D @ T for i, T in enumerate(poses)]
    else:
        poses = [look_at(np.array([BARREL[0] + 3 * np.cos(a), BARREL[1] + 3 * np.sin(a), 1.5]), target)
                 for a in np.linspace(-np.pi * 0.9, np.pi * 0.1, 8)]
        recorded = poses
    for i, (T, T_rec) in enumerate(zip(poses, recorded)):
        img, _ = render(T, IMG_W, IMG_H, K)
        _, depth = render(T, DEPTH_W, DEPTH_H, kd)
        conf = np.full((DEPTH_H, DEPTH_W), 2, dtype=np.uint8)
        stem = f"keyframes/{i:05d}"
        Image.fromarray(img).save(root / f"{stem}.heic")
        depth.astype("<f4").tofile(root / f"{stem}.depth.f32")
        conf.tofile(root / f"{stem}.conf.u8")
        entries.append({"index": i, "timestamp": float(i), "image": f"{stem}.heic", "depth": f"{stem}.depth.f32",
                        "confidence": f"{stem}.conf.u8", "image_size": [IMG_W, IMG_H], "depth_size": [DEPTH_W, DEPTH_H],
                        "intrinsics": K.ravel().tolist(), "camera_to_world": T_rec.ravel().tolist()})
    (root / "poses.json").write_text(json.dumps({"keyframes": entries}, indent=1))

    manifest = {
        "format": "scan2sim-bundle", "version": 1, "name": "fake-driveway", "units": "meters", "frame": "z_up_right_handed",
        "mesh": {"file": "mesh.glb", "parts": []},
        "objects": [{"name": "barrel", "file": "objects/barrel.glb", "tag_center": [*BARREL, base_z + 0.45], "tag_radius_m": 0.6}],
        "keyframes": {"count": len(entries), "poses": "poses.json", "dir": "keyframes/"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(root)
    return root


if __name__ == "__main__":
    main(Path(sys.argv[1]), path_mode="--path" in sys.argv)
