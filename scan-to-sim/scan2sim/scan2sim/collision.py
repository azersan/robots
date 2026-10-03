"""Collision shapes for tagged objects.

Phone scans of an object are usually open shells (the side you walked past, no bottom), so the
default is the convex hull of the object's points plus their projection onto the ground: a solid
that sits on the heightfield. CoACD is used for watertight scans, or when forced, to keep concave
shapes (a chair, an open crate) as several convex pieces.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .bundle import Part
from .ground import Heightfield


def object_collision(part: Part, hf: Heightfield, method: str = "auto", coacd_threshold: float = 0.05,
                     max_hulls: int = 16) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Returns ([(vertices, faces), ...] convex pieces in sim frame, method actually used)."""
    mesh = trimesh.Trimesh(part.vertices, part.faces, process=True)
    if method == "coacd" or (method == "auto" and mesh.is_watertight):
        try:
            import coacd

            coacd.set_log_level("error")
            pieces = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=coacd_threshold,
                                     max_convex_hull=max_hulls)
            return [(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)) for v, f in pieces], "coacd"
        except ImportError:
            print("  coacd not installed; using a hull to the ground")
    pts = mesh.vertices
    ground = pts.copy()
    ground[:, 2] = hf.sample(pts[:, 0], pts[:, 1])
    hull = trimesh.Trimesh(np.concatenate([pts, ground])).convex_hull
    return [(np.asarray(hull.vertices), np.asarray(hull.faces, dtype=np.int64))], "hull_to_ground"


def body_origin(part: Part, hf: Heightfield) -> np.ndarray:
    """Object frame origin: centroid in x/y, on the ground beneath it."""
    c = part.vertices.mean(axis=0)
    return np.array([c[0], c[1], float(hf.sample(c[0], c[1]))])


def hull_volume(v: np.ndarray, f: np.ndarray) -> float:
    return float(abs(trimesh.Trimesh(v, f, process=False).volume))


def write_obj(path, v: np.ndarray, f: np.ndarray):
    with open(path, "w") as out:
        out.write("".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in v))
        out.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f))
