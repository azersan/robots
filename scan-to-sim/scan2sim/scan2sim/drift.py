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


def depth_points(kf: Keyframe, stride: int | None = None, min_confidence: int = 1, max_depth: float = 4.0) -> np.ndarray:
    """World-space (sim frame) points from a keyframe's depth map."""
    depth = kf.load_depth()
    conf = kf.load_confidence()
    dh, dw = depth.shape
    stride = stride or max(1, dw // 64)  # ~64 samples per row (the phone's 256x192 depth -> every 4th pixel)
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


def normals(p: np.ndarray, k: int = 12) -> np.ndarray:
    """Unit normals from local PCA (sign arbitrary)."""
    _, idx = cKDTree(p).query(p, k=min(k, len(p)))
    nb = p[idx] - p[idx].mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(nb, full_matrices=False)
    return vt[:, -1, :]


def normal_space_sample(p: np.ndarray, n: np.ndarray, per_bin: int = 1500, seed: int = 0) -> np.ndarray:
    """Indices sampled evenly across normal directions, so walls and curbs weigh as much as the ground."""
    rng = np.random.default_rng(seed)
    nz = np.abs(n[:, 2])
    az = np.floor((np.arctan2(n[:, 1], n[:, 0]) % np.pi) / (np.pi / 6)).astype(int)  # 6 horizontal directions
    bins = np.where(nz > 0.8, 6, np.where(nz < 0.5, az, 7))
    out = []
    for b in np.unique(bins):
        idx = np.nonzero(bins == b)[0]
        out.append(rng.choice(idx, min(per_bin, len(idx)), replace=False))
    return np.concatenate(out)


def icp(source: np.ndarray, target: np.ndarray, iterations: int = 60, max_pair: float = 0.8,
        init: np.ndarray | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    """Rigid transform (4x4) moving source onto target (point-to-plane, normal-space sampled), and the median
    point-to-plane residual."""
    tn = normals(target)
    sn = normals(source)
    src = source[normal_space_sample(source, sn)]
    tree = cKDTree(target)
    T = np.eye(4) if init is None else init.copy()
    for it in range(iterations):
        cur = (T[:3, :3] @ src.T).T + T[:3, 3]
        dist, j = tree.query(cur)
        limit = max(max_pair * (0.9 ** it), 0.08)
        keep = dist < limit
        if keep.sum() < 30:
            break
        a, q, n = cur[keep], target[j[keep]], tn[j[keep]]
        # Center the rotation on the points so rotation and translation are well separated.
        c = a.mean(axis=0)
        A = np.column_stack([np.cross(a - c, n), n])
        b = ((q - a) * n).sum(axis=1)
        H = A.T @ A
        # Damped (Levenberg-style): directions the geometry doesn't constrain stay put instead of running off.
        x = np.linalg.solve(H + 1e-3 * np.trace(H) / 6 * np.eye(6), A.T @ b)
        w, t = x[:3], x[3:]
        t = t + c - (np.eye(3) + _skew(w)) @ c if np.linalg.norm(w) > 0 else t
        theta = np.linalg.norm(w)
        if theta > 1e-12:
            kx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / theta
            R = np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * kx @ kx
        else:
            R = np.eye(3)
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, t
        T = step @ T
        if theta < 1e-6 and np.linalg.norm(t) < 1e-5 and limit <= 0.08:
            break
    cur = (T[:3, :3] @ src.T).T + T[:3, 3]
    dist, j = tree.query(cur)
    keep = dist < 0.1
    resid = np.abs(((cur[keep] - target[j[keep]]) * tn[j[keep]]).sum(axis=1)) if keep.any() else np.array([np.inf])
    # Point-to-plane information matrix at the solution, rotation scaled to meters at the cloud's radius,
    # so its eigenvalues say how well each motion is pinned down.
    a, n = cur[keep], tn[j[keep]]
    c = a.mean(axis=0) if len(a) else np.zeros(3)
    radius = float(np.sqrt(((a - c) ** 2).sum(axis=1).mean())) if len(a) else 1.0
    A = np.column_stack([np.cross(a - c, n) / max(radius, 1e-6), n]) if len(a) else np.zeros((0, 6))
    return T, float(np.median(resid)), A.T @ A / max(len(a), 1)


MOTIONS = ["roll", "pitch", "yaw", "x", "y", "z"]


def degeneracy(H: np.ndarray, min_ratio: float = 0.01) -> list[str]:
    """Motions the geometry barely constrains (eigenvalues below min_ratio of the strongest)."""
    vals, vecs = np.linalg.eigh(H)
    weak = []
    for v, vec in zip(vals, vecs.T):
        if v < min_ratio * vals[-1]:
            weak += [MOTIONS[k] for k in np.nonzero(np.abs(vec) > 0.5)[0]] or [MOTIONS[int(np.argmax(np.abs(vec)))]]
    return sorted(set(weak), key=MOTIONS.index)


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])




def common_space(a: np.ndarray, b: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep only points in coarse cells (dilated by one) that both clouds occupy."""
    ka = {tuple(k) for k in np.floor(a / cell).astype(np.int64)}
    kb = {tuple(k) for k in np.floor(b / cell).astype(np.int64)}
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    both = {(x + i, y + j, z + k) for (x, y, z) in ka & {(x + i, y + j, z + k) for (x, y, z) in kb for i, j, k in offsets}
            for i, j, k in offsets}

    def keep(p):
        cells = np.floor(p / cell).astype(np.int64)
        return p[np.array([tuple(c) in both for c in cells], dtype=bool)] if len(p) else p

    return keep(a), keep(b)


def loop_drift(keyframes: list[Keyframe], radius: float = 3.0, voxel: float = 0.05, force_overlap: bool = False) -> dict:
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
    method = "matched views at the start"
    if out_idx and ret_idx and not force_overlap:
        out_pts = voxel_down(np.concatenate([depth_points(keyframes[i]) for i in out_idx]), voxel)
        ret_pts = voxel_down(np.concatenate([depth_points(keyframes[i]) for i in ret_idx]), voxel)
        ref = start
    else:
        # Fallback: the end of the walk vs the outbound trip, wherever they saw the same space, in any direction.
        method = "overlap of the walk's end with the outbound trip"
        tail = list(range(int(len(keyframes) * 0.85), len(keyframes)))
        end_pos = pos[tail].mean(axis=0)
        far_i = int(np.argmax(np.linalg.norm(pos[:, :2] - start[:2], axis=1)))
        outbound = [i for i in range(far_i) if np.linalg.norm(pos[i, :2] - end_pos[:2]) < 8.0]
        if not outbound:
            return {**base, "revisited": False,
                    "note": "the walk did not come back to ground seen on the way out; end near the start to measure drift"}
        out_pts = voxel_down(np.concatenate([depth_points(keyframes[i]) for i in outbound]), voxel)
        ret_pts = voxel_down(np.concatenate([depth_points(keyframes[i]) for i in tail]), voxel)
        out_pts, ret_pts = common_space(out_pts, ret_pts, cell=0.5)
        ref = end_pos
        out_idx, ret_idx = outbound, tail
    if len(out_pts) < 200 or len(ret_pts) < 200:
        return {**base, "revisited": False, "method": method, "note": "too little shared depth to align"}
    T, residual, H = icp(ret_pts, out_pts)
    weak = degeneracy(H, min_ratio=0.005)

    # Robustness: start from deliberately wrong guesses. If they don't converge to the same answer, the shared
    # geometry doesn't determine the drift, and the spread is an honest error bar.
    def drift_at_ref(T_):
        E_ = np.linalg.inv(T_)
        return E_[:3, :3] @ ref + E_[:3, 3] - ref

    c = ret_pts.mean(axis=0)
    answers = [drift_at_ref(T)]
    for dx, dy, yaw in [(0.3, 0, 0), (-0.3, 0, 0), (0, 0.3, 0), (0, -0.3, 0), (0, 0, 3), (0, 0, -3)]:
        a = np.radians(yaw)
        init = np.eye(4)
        init[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        init[:3, 3] = c - init[:3, :3] @ c + [dx, dy, 0]
        answers.append(drift_at_ref(icp(ret_pts, out_pts, init=init)[0]))
    answers = np.array(answers)
    spread = float(np.linalg.norm(answers - np.median(answers, axis=0), axis=1).max())
    # T moves the return cloud onto the outbound one, so the return poses' error is its inverse.
    E = np.linalg.inv(T)
    # Drift expressed at the start point, not at the sim origin.
    moved = E[:3, :3] @ ref + E[:3, 3] - ref
    angle = float(np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))))
    yaw = float(np.degrees(np.arctan2(E[1, 0], E[0, 0])))
    drift = float(np.linalg.norm(moved))
    return {
        **base,
        "revisited": True,
        "method": method,
        "weak_axes": weak,
        "uncertainty_m": round(spread, 3),
        "keyframes_outbound": len(out_idx),
        "keyframes_return": len(ret_idx),
        "drift_m": round(drift, 3),
        "drift_xyz_m": [round(float(x), 3) for x in moved],  # where the return trip thought the start was, minus where it is
        "rotation_deg": round(angle, 2),
        "yaw_deg": round(yaw, 2),
        "drift_percent_of_path": round(100 * drift / max(path_length, 1e-6), 2),
        "alignment_residual_m": round(residual, 3),
        "verdict": ("unreliable: the end of the walk and the start share mostly smooth ground, which doesn't pin down "
                    "the drift; finish facing walls, posts or edges you saw at the start"
                    if spread > max(0.1, 0.3 * drift) or weak else
                    "good" if drift < 0.2 else "usable, expect some ghosting" if drift < 0.5
                    else "high: consider splitting into overlapping scans"),
    }
