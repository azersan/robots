"""Scanned ground for a site: a heightfield in the world frame, from tools/import_scan.py.

The grid is axis-aligned (X east, Y north, meters). Heights are clamped at the grid edge, so anything
placed outside it sits at the nearest edge height. `measured` marks cells the phone scan actually saw;
the rest is a smooth fill.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton

TERRAIN_DIR = Path(__file__).resolve().parents[1] / "terrain"


class Terrain:
    def __init__(self, x0, y0, cell, z, measured=None, meta=None):
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)
        self.z = np.asarray(z, dtype=np.float32)
        self.measured = measured if measured is not None else np.ones(self.z.shape, bool)
        self.meta = meta or {}

    @classmethod
    def load(cls, site):
        path = TERRAIN_DIR / f"{site}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; create it with tools/import_scan.py")
        d = np.load(path)
        return cls(d["x0"], d["y0"], d["cell"], d["z"], d["measured"], json.loads(str(d["meta"])))

    @property
    def bounds(self):
        ny, nx = self.z.shape
        return self.x0, self.y0, self.x0 + (nx - 1) * self.cell, self.y0 + (ny - 1) * self.cell

    def height(self, x, y):
        """Bilinear ground height at (x, y); scalars or arrays."""
        ny, nx = self.z.shape
        fx = np.clip((np.asarray(x, np.float64) - self.x0) / self.cell, 0, nx - 1)
        fy = np.clip((np.asarray(y, np.float64) - self.y0) / self.cell, 0, ny - 1)
        ix = np.minimum(fx.astype(int), nx - 2)
        iy = np.minimum(fy.astype(int), ny - 2)
        tx, ty = fx - ix, fy - iy
        z = self.z
        h = (z[iy, ix] * (1 - tx) + z[iy, ix + 1] * tx) * (1 - ty) + (z[iy + 1, ix] * (1 - tx) + z[iy + 1, ix + 1] * tx) * ty
        return float(h) if np.ndim(h) == 0 else h

    def support(self, x, y, yaw, half_x, half_y):
        """Highest ground under a rectangle footprint (center, 4 corners, edge midpoints): spawn height
        that never starts an object inside a slope."""
        c, s = np.cos(yaw), np.sin(yaw)
        lx = np.array([0, 1, 1, -1, -1, 1, -1, 0, 0]) * half_x
        ly = np.array([0, 1, -1, 1, -1, 0, 0, 1, -1]) * half_y
        return float(np.max(self.height(x + c * lx - s * ly, y + s * lx + c * ly)))

    def add_physics(self, builder, cfg):
        """The ground collider. Newton's CollisionPipeline handles heightfields exactly (MuJoCo Warp's own
        heightfield contacts were ~10-15 cm high on scanned terrain, so the sim must keep
        use_mujoco_contacts=False)."""
        ny, nx = self.z.shape
        hx, hy = (nx - 1) * self.cell / 2, (ny - 1) * self.cell / 2
        hf = newton.Heightfield(data=self.z, nrow=ny, ncol=nx, hx=hx, hy=hy,
                                min_z=float(self.z.min()), max_z=float(self.z.max()) + 1e-3)
        builder.add_shape_heightfield(xform=wp.transform(p=wp.vec3(self.x0 + hx, self.y0 + hy, 0.0)),
                                      heightfield=hf, cfg=cfg, label="ground")

    def drape(self, verts, lift=0.0):
        """Raise mesh vertices (N x 3, built at z = 0 plus their own offset) onto the ground."""
        v = np.array(verts, np.float64)
        v[:, 2] += self.height(v[:, 0], v[:, 1]) + lift
        return v
