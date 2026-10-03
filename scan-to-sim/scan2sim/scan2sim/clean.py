"""Mesh cleanup: weld duplicate vertices, drop degenerate faces and tiny floating fragments, optional decimation."""
from __future__ import annotations

import numpy as np
import trimesh

from .bundle import Part


def clean(part: Part, min_component_faces: int = 30, max_faces: int | None = None) -> Part:
    if len(part.faces) == 0:
        return part
    m = trimesh.Trimesh(part.vertices, part.faces, process=True, validate=True)
    if min_component_faces > 0 and len(m.faces) > min_component_faces:
        labels = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(m.faces))
        counts = np.bincount(labels)
        keep = counts[labels] >= min_component_faces
        if keep.any() and not keep.all():
            m.update_faces(keep)
            m.remove_unreferenced_vertices()
    if max_faces and len(m.faces) > max_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=max_faces)
        except Exception as e:  # fast-simplification missing or failed: keep full resolution
            print(f"  decimation skipped for {part.name}: {e}")
    return Part(name=part.name, vertices=np.asarray(m.vertices), faces=np.asarray(m.faces, dtype=np.int64), meta=part.meta)
