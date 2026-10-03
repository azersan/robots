"""Find bins in the front camera's depth image, without using their true poses.

Depth pixels -> world points (using the car pose and the camera mount) -> keep points
15 cm to 1.2 m above the ground -> grid clustering in XY -> keep clusters with a
bin-sized footprint. Color-agnostic on purpose: real bins can be any color.
"""

import io
import math

import numpy as np
from PIL import Image


def _rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def camera_rays(width, height, fov_deg):
    """Unit ray per pixel in the camera frame (OpenGL: -Z forward, +Y up, +X right)."""
    f = (height / 2) / math.tan(math.radians(fov_deg) / 2)
    u = (np.arange(width) + 0.5 - width / 2) / f
    v = -(np.arange(height) + 0.5 - height / 2) / f
    uu, vv = np.meshgrid(u, v)
    d = np.stack([uu, vv, -np.ones_like(uu)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def camera_basis(direction):
    """Columns: camera +X (right), +Y (up), +Z (back) in the body frame."""
    f = np.asarray(direction, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, (0.0, 0.0, 1.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], axis=1)


class BinFinder:
    def __init__(self, info):
        cam = info["cameras"]["front"]
        mount = info["front_mount"]
        self.rays = camera_rays(cam["width"], cam["height"], cam["fov"])
        self.R_body_cam = camera_basis(mount["dir"])
        self.t_body_cam = np.asarray(mount["pos"])
        self.chassis_z = mount["chassis_z"]
        self.half = info["bin"]["half_extents"]

    def points(self, depth_png, car):
        """World XYZ for every valid depth pixel. Assumes the car is level (flat ground)."""
        d = np.asarray(Image.open(io.BytesIO(depth_png)), dtype=np.float32) / 1000.0
        valid = (d > 0.05) & (d < 12.0)
        p_cam = self.rays[valid] * d[valid, None]
        R = _rot_z(car["yaw"])
        p_body = p_cam @ self.R_body_cam.T + self.t_body_cam
        return p_body @ R.T + np.array([car["x"], car["y"], self.chassis_z])

    def find(self, depth_png, car, cell=0.1):
        pts = self.points(depth_png, car)
        band = pts[(pts[:, 2] > 0.15) & (pts[:, 2] < 1.2)]
        if len(band) < 30:
            return []
        # Grid clustering: occupied cells, then 8-connected components.
        ij = np.floor(band[:, :2] / cell).astype(int)
        cells = {}
        for k, key in enumerate(map(tuple, ij)):
            cells.setdefault(key, []).append(k)
        seen, found = set(), []
        for start in cells:
            if start in seen:
                continue
            stack, members = [start], []
            seen.add(start)
            while stack:
                c = stack.pop()
                members.extend(cells[c])
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        n = (c[0] + dx, c[1] + dy)
                        if n in cells and n not in seen:
                            seen.add(n)
                            stack.append(n)
            cluster = band[members]
            det = self._describe(cluster, car)
            if det:
                found.append(det)
        found.sort(key=lambda b: b["range"])
        return found

    def _describe(self, cluster, car):
        if len(cluster) < 40:
            return None
        if cluster[:, 2].max() < 0.7:  # bins are ~1.1 m tall
            return None
        xy = cluster[:, :2]
        size = xy.max(axis=0) - xy.min(axis=0)
        # Bins are ~0.6-0.75 m a side; trunks are ~0.3 m, houses and hedges much wider.
        if not 0.4 < float(np.hypot(*size)) < 1.3:
            return None
        fit = self._fit_box(xy, np.array([car["x"], car["y"]]))
        if fit is None:
            return None
        center, normal, extents = fit
        face_mid = center + normal * self.half[0]
        rng = float(np.linalg.norm(face_mid - [car["x"], car["y"]]))
        return {
            "center": center.tolist(),
            "face_mid": face_mid.tolist(),  # center of the bin's front face, on the ground plane
            "yaw": math.atan2(normal[1], normal[0]),  # front normal (same convention as a bin's yaw)
            "extents": extents,
            "height": float(cluster[:, 2].max()),
            "range": rng,
            "points": int(len(cluster)),
        }

    def _fit_box(self, xy, car_xy):
        """Fit the bin's known footprint to the visible points (usually one face, or an L of two).

        Orientation from the minimum-area bounding rectangle; the longer side is the bin's
        depth axis (front-to-back), and the front is taken to be the narrow face toward the car.
        """
        depth_len, width_len = 2 * self.half[0], 2 * self.half[1]
        best = None
        for deg in range(0, 90):
            t = math.radians(deg)
            e1 = np.array([math.cos(t), math.sin(t)])
            e2 = np.array([-e1[1], e1[0]])
            a, b = xy @ e1, xy @ e2
            area = (a.max() - a.min()) * (b.max() - b.min())
            if best is None or area < best[0]:
                best = (area, e1, e2, a, b)
        _, e1, e2, a, b = best
        ext = [float(a.max() - a.min()), float(b.max() - b.min())]
        threshold = (depth_len + width_len) / 2
        if max(ext) >= threshold:
            depth_axis = 0 if ext[0] >= ext[1] else 1  # we can see the full long side
        else:
            depth_axis = 1 if ext[0] >= ext[1] else 0  # we see a narrow face head-on
        axes, proj = (e1, e2), (a, b)
        center = np.zeros(2)
        for k in range(2):
            length = depth_len if k == depth_axis else width_len
            p, c = proj[k], car_xy @ axes[k]
            mid = (p.min() + p.max()) / 2
            # The visible face is on the car's side; the far side is hidden, so place it by known size.
            coord = p.min() + length / 2 if c < mid else p.max() - length / 2
            center += coord * axes[k]
        normal = axes[depth_axis] * (1.0 if (car_xy - center) @ axes[depth_axis] > 0 else -1.0)
        return center, normal, ext
