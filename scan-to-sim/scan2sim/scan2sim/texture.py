"""Per-vertex colors from the keyframes.

Each vertex takes its color from the closest keyframe that actually sees it: the vertex must
project inside the image and agree with that keyframe's LiDAR depth, using only depth pixels at or
above the confidence threshold (bright sun lowers confidence outdoors). Vertex colors are the v1
texture; a UV atlas can replace this later without changing the bundle.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .bundle import Keyframe, Part

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC keyframes then fail to open with a clear PIL error
    pass

DEFAULT_COLOR = np.array([0.6, 0.6, 0.6], dtype=np.float32)


def colorize(parts: list[Part], keyframes: list[Keyframe], min_confidence: int = 1, max_distance: float = 6.0,
             image_scale: float = 0.5) -> dict:
    """Fills part.colors in place. Returns stats: per-part fraction of vertices seen by any keyframe."""
    offsets = np.cumsum([0] + [len(p.vertices) for p in parts])
    v = np.concatenate([p.vertices for p in parts])
    best = np.full(len(v), np.inf)  # camera distance of the keyframe each vertex took its color from
    colors = np.tile(DEFAULT_COLOR, (len(v), 1))
    homo = np.column_stack([v, np.ones(len(v))])

    for kf in keyframes:
        img = Image.open(kf.image).convert("RGB")
        if image_scale != 1:
            img = img.resize((max(1, int(img.width * image_scale)), max(1, int(img.height * image_scale))), Image.BILINEAR)
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        ih, iw = rgb.shape[:2]
        sx, sy = iw / kf.image_size[0], ih / kf.image_size[1]
        depth = kf.load_depth()
        conf = kf.load_confidence()
        dh, dw = depth.shape

        pc = (np.linalg.inv(kf.cam_to_world) @ homo.T).T[:, :3]
        d = -pc[:, 2]  # distance along the view axis (camera looks along -z)
        ok = (d > 0.1) & (d < max_distance)
        idx = np.nonzero(ok & (d < best))[0]
        if len(idx) == 0:
            continue
        x, y, d_sel = pc[idx, 0], pc[idx, 1], d[idx]
        K = kf.K
        u = K[0, 0] * (x / d_sel) + K[0, 2]  # image y points down, camera +y points up
        w = K[1, 1] * (-y / d_sel) + K[1, 2]
        inside = (u >= 0) & (u < kf.image_size[0]) & (w >= 0) & (w < kf.image_size[1])
        idx, u, w, d_sel = idx[inside], u[inside], w[inside], d_sel[inside]

        du = np.clip((u * dw / kf.image_size[0]).astype(int), 0, dw - 1)
        dv = np.clip((w * dh / kf.image_size[1]).astype(int), 0, dh - 1)
        visible = np.abs(depth[dv, du] - d_sel) < 0.05 + 0.02 * d_sel
        if conf is not None:
            visible &= conf[dv, du] >= min_confidence
        idx, u, w, d_sel = idx[visible], u[visible], w[visible], d_sel[visible]

        pu = np.clip((u * sx).astype(int), 0, iw - 1)
        pv = np.clip((w * sy).astype(int), 0, ih - 1)
        colors[idx] = rgb[pv, pu]
        best[idx] = d_sel

    stats = {}
    for p, a, b in zip(parts, offsets[:-1], offsets[1:]):
        p.colors = colors[a:b].astype(np.float32)
        stats[p.name] = float(np.isfinite(best[a:b]).mean()) if b > a else 0.0
    return stats
