"""Loop-closure drift: how far the camera poses moved between leaving the start area and coming back.

The keyframes near the start point are split into the outbound visit (first half of the sequence) and the
return visit (second half). Each visit's LiDAR depth is turned into a point cloud using its recorded poses,
and the return cloud is aligned to the outbound cloud with ICP. If the poses had no drift the two clouds
already coincide; the ICP correction is the drift.

ARKit may correct its mesh after it recognizes the start area, but recorded keyframe poses are never
corrected afterwards, so this measures the raw drift of the pose stream (and how far off keyframe colors
from the far end of the path can be).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .bundle import Keyframe


def depth_points(kf: Keyframe, stride: int = 4, min_confidence: int = 1, max_depth: float = 4.0) -> np.ndarray:
    """World-space (sim frame) points from a keyframe's depth map."""
    depth = kf.load_depth()
    conf = kf.load_confidence()
    dh, dw = depth.shape
    sx, sy = dw / kf.image_size[0], dh / kf.image_size[1]
    fx, fy, cx, cy = kf.K[0, 0] * sx, kf.K[1, 1] * sy, kf.K[0, 2] * sx, kf.K[1, 2] * sy
    v, u = np.mgrid[0:dh:stride, 0:dw:stride]
    d = depth[v, u]
    ok = (d > 0.1) & (d < max_depth)
    if conf is not None:
        ok &= conf[v, u] >= min_confidence
    u, v, d = u[ok] + 0.5, v[ok] + 0.5, d[ok]
    cam = np.column_stack([(u - cx) / fx * d, -(v - cy) / fy * d, -d, np.ones_like(d)])
    return (kf.cam_to_world @ cam.T).T[:, :3]


def voxel_down(p: np.ndarray, size: float) -> np.ndarray:
    if len(p) == 0:
        return p
    _, idx = np.unique(np.floor(p / size).astype(np.int64), axis=0, return_index=True)
    return p[idx]


def icp(source: np.ndarray, target: np.ndarray, iterations: int = 40, max_pair: float = 0.5) -> tuple[np.ndarray, float]:
    """Rigid transform (4x4) moving source onto target, and the final median pair distance."""
    tree = cKDTree(target)
    T = np.eye(4)
    src = source.copy()
    # Start with the centroid offset of overlapping regions so large drifts still converge.
    for it in range(iterations):
        dist, j = tree.query(src)
        limit = max(max_pair * (0.85 ** it), 0.05)
        keep = dist < limit
        if keep.sum() < 20:
            break
        a, b = src[keep], target[j[keep]]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        R = (U @ Vt).T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = (U @ Vt).T
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, cb - R @ ca
        src = (step[:3, :3] @ src.T).T + step[:3, 3]
        T = step @ T
    dist, _ = tree.query(src)
    return T, float(np.median(dist))


def loop_drift(keyframes: list[Keyframe], radius: float = 3.0, voxel: float = 0.05) -> dict:
    if len(keyframes) < 4:
        return {"revisited": False, "note": "too few keyframes"}
    pos = np.array([k.cam_to_world[:3, 3] for k in keyframes])
    start = pos[0]
    near = np.linalg.norm(pos[:, :2] - start[:2], axis=1) < radius
    half = len(keyframes) // 2
    out_idx = [i for i in range(half) if near[i]]
    ret_idx = [i for i in range(half, len(keyframes)) if near[i]]
    # Only compare views of the same scene: return keyframes within 1.5 m of an outbound keyframe and looking
    # the same way (within 35 deg). Otherwise the clouds barely overlap and ICP slides along the ground.
    fwd = np.array([-k.cam_to_world[:3, 2] for k in keyframes])
    pairs = [(i, j) for i in out_idx for j in ret_idx
             if np.linalg.norm(pos[i] - pos[j]) < 1.5 and fwd[i] @ fwd[j] > np.cos(np.radians(35))]
    out_idx = sorted({i for i, _ in pairs})
    ret_idx = sorted({j for _, j in pairs})
    path_length = float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum())
    far = float(np.linalg.norm(pos[:, :2] - start[:2], axis=1).max())
    base = {"path_length_m": round(path_length, 2), "farthest_from_start_m": round(far, 2)}
    if not out_idx or not ret_idx:
        return {**base, "revisited": False,
                "note": f"no return view of the start: finish within {radius} m of the start, facing the way you set off"}

    out_pts = voxel_down(np.concatenate([depth_points(keyframes[i]) for i in out_idx]), voxel)
    ret_pts = voxel_down(np.concatenate([depth_points(keyframes[i]) for i in ret_idx]), voxel)
    if len(out_pts) < 100 or len(ret_pts) < 100:
        return {**base, "revisited": True, "note": "too little depth near the start to align"}
    T, residual = icp(ret_pts, out_pts)
    # T moves the return cloud onto the outbound one, so the return poses' error is its inverse.
    E = np.linalg.inv(T)
    # Drift expressed at the start point, not at the sim origin.
    moved = E[:3, :3] @ start + E[:3, 3] - start
    angle = float(np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))))
    yaw = float(np.degrees(np.arctan2(E[1, 0], E[0, 0])))
    drift = float(np.linalg.norm(moved))
    return {
        **base,
        "revisited": True,
        "keyframes_outbound": len(out_idx),
        "keyframes_return": len(ret_idx),
        "drift_m": round(drift, 3),
        "drift_xyz_m": [round(float(x), 3) for x in moved],  # where the return trip thought the start was, minus where it is
        "rotation_deg": round(angle, 2),
        "yaw_deg": round(yaw, 2),
        "drift_percent_of_path": round(100 * drift / max(path_length, 1e-6), 2),
        "alignment_residual_m": round(residual, 3),
        "verdict": ("good" if drift < 0.2 else "usable, expect some ghosting" if drift < 0.5
                    else "high: consider splitting into overlapping scans"),
    }
