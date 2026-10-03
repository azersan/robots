"""Finding the ground geometrically: a heightfield of the lowest upward-facing surfaces.

ARKit's per-face classes are not trusted here (outdoors, grass and asphalt are often "none").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .bundle import Part


@dataclass
class Heightfield:
    x0: float  # center of cell [0, 0]
    y0: float
    cell: float
    z: np.ndarray  # (ny, nx); row = y index, col = x index
    measured: np.ndarray  # (ny, nx) bool, cells that had scan data (others are filled)

    @property
    def shape(self):
        return self.z.shape

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear height at (x, y); clamps outside the grid."""
        ny, nx = self.z.shape
        fx = np.clip((np.asarray(x) - self.x0) / self.cell, 0, nx - 1)
        fy = np.clip((np.asarray(y) - self.y0) / self.cell, 0, ny - 1)
        ix = np.minimum(fx.astype(int), nx - 2) if nx > 1 else np.zeros_like(fx, dtype=int)
        iy = np.minimum(fy.astype(int), ny - 2) if ny > 1 else np.zeros_like(fy, dtype=int)
        tx, ty = fx - ix, fy - iy
        z = self.z
        if nx == 1 or ny == 1:
            return z[iy, ix]
        return ((z[iy, ix] * (1 - tx) + z[iy, ix + 1] * tx) * (1 - ty)
                + (z[iy + 1, ix] * (1 - tx) + z[iy + 1, ix + 1] * tx) * ty)

    def downsample(self, factor: int) -> "Heightfield":
        if factor <= 1:
            return self
        return Heightfield(self.x0, self.y0, self.cell * factor, self.z[::factor, ::factor].copy(),
                           self.measured[::factor, ::factor].copy())

    def to_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Triangle mesh of the grid (upward-facing, CCW seen from +Z)."""
        ny, nx = self.z.shape
        xs = self.x0 + np.arange(nx) * self.cell
        ys = self.y0 + np.arange(ny) * self.cell
        gx, gy = np.meshgrid(xs, ys)
        verts = np.column_stack([gx.ravel(), gy.ravel(), self.z.ravel()])
        i = np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)[None, :]
        i = i.ravel()
        faces = np.concatenate([np.column_stack([i, i + 1, i + nx + 1]), np.column_stack([i, i + nx + 1, i + nx])])
        return verts, faces


def face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(length, 1e-12)


def fit_heightfield(parts: list[Part], cell: float = 0.05, min_up: float = 0.8, margin: float = 0.25,
                    max_rise: float = 0.3) -> Heightfield:
    """Per cell, the lowest point on an upward-facing surface; spikes and raised surfaces (tabletops, seats)
    removed; holes filled from neighbors.

    ARKit winds faces consistently (floor normals point up, ceilings down), so only up-facing faces count.
    """
    v = np.concatenate([p.vertices for p in parts])
    f = np.concatenate([p.faces + off for p, off in zip(parts, np.cumsum([0] + [len(p.vertices) for p in parts[:-1]]))])
    n = face_normals(v, f)
    up = n[:, 2] >= min_up
    if not up.any():
        raise ValueError("no near-horizontal surfaces found; cannot fit ground")

    # Sample up-facing triangles at vertices and centroids.
    tri = v[f[up]]
    pts = np.concatenate([tri.reshape(-1, 3), tri.mean(axis=1)])

    lo = v[:, :2].min(axis=0) - margin
    hi = v[:, :2].max(axis=0) + margin
    nx = int(np.ceil((hi[0] - lo[0]) / cell)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / cell)) + 1
    ix = np.clip(np.round((pts[:, 0] - lo[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(np.round((pts[:, 1] - lo[1]) / cell).astype(int), 0, ny - 1)
    z = np.full((ny, nx), np.inf)
    np.minimum.at(z, (iy, ix), pts[:, 2])
    measured = np.isfinite(z)

    # Where the floor under a table or seat wasn't seen, the lowest up-facing surface is the top of the
    # furniture. Reject cells more than max_rise above the local ground level (low percentile of a
    # coarse 0.25 m grid over a ~2 m window).
    measured &= ~_raised(z, measured, cell, max_rise)

    # Remove isolated low spikes (LiDAR noise): a measured cell far below its neighborhood median.
    filled = _fill(np.where(measured, z, np.inf), measured)
    med = ndimage.median_filter(filled, size=5)
    spikes = measured & (filled < med - 0.1)
    measured &= ~spikes
    z = _fill(np.where(measured, z, np.inf), measured)
    z = ndimage.median_filter(z, size=3)
    return Heightfield(lo[0], lo[1], cell, z, measured)


def _raised(z: np.ndarray, measured: np.ndarray, cell: float, max_rise: float) -> np.ndarray:
    block = max(1, int(round(0.25 / cell)))
    ny, nx = z.shape
    py, px = -ny % block, -nx % block
    zz = np.pad(np.where(measured, z, np.inf), ((0, py), (0, px)), constant_values=np.inf)
    coarse = zz.reshape(zz.shape[0] // block, block, zz.shape[1] // block, block).min(axis=(1, 3))
    valid = np.isfinite(coarse)
    if not valid.any():
        return np.zeros_like(measured)
    coarse = _fill(coarse, valid)
    ref = ndimage.percentile_filter(coarse, 20, size=9)
    ref = np.repeat(np.repeat(ref, block, axis=0), block, axis=1)[:ny, :nx]
    return measured & (z > ref + max_rise)


def _fill(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid cells with the nearest valid value."""
    if valid.all():
        return z.copy()
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def split_ground(part: Part, hf: Heightfield, tol: float = 0.06, min_up: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (ground, rest) over the part's faces: ground = near-horizontal and on the heightfield."""
    v, f = part.vertices, part.faces
    n = face_normals(v, f)
    dz = np.abs(v[:, 2] - hf.sample(v[:, 0], v[:, 1]))
    on = (dz[f] <= tol).all(axis=1) & (n[:, 2] >= min_up)
    return on, ~on


def submesh(part: Part, mask: np.ndarray, name: str) -> Part:
    """Faces selected by mask, with unused vertices dropped."""
    faces = part.faces[mask]
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    colors = part.colors[used] if part.colors is not None else None
    return Part(name=name, vertices=part.vertices[used], faces=inverse.reshape(-1, 3), colors=colors, meta=dict(part.meta))


def ghost_layer(part: Part, hf: Heightfield, low: float = 0.03, high: float = 0.25, min_flat: float = 0.9) -> np.ndarray:
    """Face mask of a duplicate ground skin: flat faces (either winding) floating 3-25 cm above measured ground.

    When two passes disagree about the ground height (drift), ARKit's fused mesh keeps both surfaces, and the
    gap between them gets a downward-facing underside. The heightfield already holds the lowest surface; these
    faces are the extra skin. Real raised surfaces (a curb top) are themselves the lowest surface in their
    cells, so they sit on the heightfield and are not caught here.
    """
    v, f = part.vertices, part.faces
    n = face_normals(v, f)
    c = v[f].mean(axis=1)
    dz = c[:, 2] - hf.sample(c[:, 0], c[:, 1])
    ny, nx = hf.z.shape
    ix = np.clip(np.round((c[:, 0] - hf.x0) / hf.cell).astype(int), 0, nx - 1)
    iy = np.clip(np.round((c[:, 1] - hf.y0) / hf.cell).astype(int), 0, ny - 1)
    return (np.abs(n[:, 2]) >= min_flat) & (dz > low) & (dz < high) & hf.measured[iy, ix]


def _cell_low(points: np.ndarray, hf: Heightfield, pct: float = 0.2) -> np.ndarray:
    """Per-cell low percentile of point heights on hf's grid (NaN where empty)."""
    ny, nx = hf.z.shape
    ix = np.round((points[:, 0] - hf.x0) / hf.cell).astype(np.int64)
    iy = np.round((points[:, 1] - hf.y0) / hf.cell).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[ok] * nx + ix[ok]
    zs = points[ok, 2]
    order = np.lexsort((zs, flat))
    flat, zs = flat[order], zs[order]
    out = np.full(ny * nx, np.nan)
    if len(flat) == 0:
        return out.reshape(ny, nx)
    starts = np.r_[0, np.nonzero(np.diff(flat))[0] + 1]
    counts = np.diff(np.r_[starts, len(flat)])
    out[flat[starts]] = zs[starts + (counts * pct).astype(np.int64)]
    return out.reshape(ny, nx)


def heightfield_from_depth(keyframes, mesh_hf: Heightfield, min_confidence: int = 1, max_rise: float = 0.3,
                           align_sigma: float = 1.0) -> tuple[Heightfield, dict]:
    """Ground from keyframe LiDAR depth, consistent with a single pass.

    Drift makes the outbound and return passes disagree about height, and ARKit's fused mesh keeps both; taking
    the lowest surface then jumps between passes and creates phantom steps. Here the outbound pass (keyframes up
    to the farthest point from the start) defines the ground; return-pass depth fills cells the outbound pass
    didn't see, after removing the local height offset measured where both passes overlap (smoothed over
    `align_sigma` meters). Cells neither pass saw keep the mesh-based height.
    """
    from .drift import depth_points

    pos = np.array([k.cam_to_world[:3, 3] for k in keyframes])
    far = int(np.argmax(np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1)))
    passes = [keyframes[: far + 1], keyframes[far + 1 :]]
    grids = []
    for frames in passes:
        pts = [depth_points(k, min_confidence=min_confidence) for k in frames]
        grids.append(_cell_low(np.concatenate(pts), mesh_hf) if pts else np.full(mesh_hf.z.shape, np.nan))
    out, ret = grids

    # Drop raised surfaces (tabletops, car hoods) the same way the mesh fit does.
    for g in (out, ret):
        valid = np.isfinite(g)
        if valid.any():
            g[_raised(np.where(valid, g, np.inf), valid, mesh_hf.cell, max_rise)] = np.nan

    both = np.isfinite(out) & np.isfinite(ret)
    stats = {"outbound_cells": int(np.isfinite(out).sum()), "return_cells": int(np.isfinite(ret).sum()),
             "overlap_cells": int(both.sum())}
    if both.any():
        sigma = align_sigma / mesh_hf.cell
        d = np.where(both, ret - out, 0.0)
        num = ndimage.gaussian_filter(d, sigma)
        den = ndimage.gaussian_filter(both.astype(float), sigma)
        have = den > 1e-3
        offset = _fill(np.where(have, num / np.maximum(den, 1e-9), 0.0), have)
        stats["return_offset_m"] = {"median": round(float(np.median(d[both])), 3),
                                    "p95_abs": round(float(np.percentile(np.abs(d[both]), 95)), 3)}
        ret = ret - offset
    z = np.where(np.isfinite(out), out, ret)
    measured = np.isfinite(z)
    z = np.where(measured, z, mesh_hf.z)
    # Remove isolated spikes and smooth lightly, as in the mesh fit.
    med = ndimage.median_filter(z, size=5)
    spikes = measured & (np.abs(z - med) > 0.1)
    z = np.where(spikes, med, z)
    z = ndimage.median_filter(z, size=3)
    stats["depth_cells"] = int(measured.sum())
    return Heightfield(mesh_hf.x0, mesh_hf.y0, mesh_hf.cell, z, mesh_hf.measured | measured), stats
