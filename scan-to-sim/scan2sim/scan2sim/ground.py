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
