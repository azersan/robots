"""Procedural visual meshes and textures for the scenes.

Everything here is visual only: physics uses simple primitives. Meshes carry
UVs (for photo textures) and per-vertex normals (for smooth shading); the
renderer multiplies a shape's color by its texture, so a white color shows the
texture as-is and a tint recolors it.
"""

import math
import os

import numpy as np
from PIL import Image, ImageFilter

import newton

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets", "textures")


# ---------------------------------------------------------------------------
# Textures


def rgba(img):
    """The renderer wants 4-channel uint8 textures."""
    if img is None or img.shape[2] == 4:
        return img
    return np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], axis=2)


def grade(img, saturation=1.0, gain=(1.0, 1.0, 1.0)):
    """Simple color grade: blend toward luminance, then per-channel gain."""
    rgb = img[..., :3].astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    rgb = lum[..., None] + saturation * (rgb - lum[..., None])
    rgb = np.clip(rgb * np.asarray(gain, np.float32), 0, 255).astype(np.uint8)
    return np.concatenate([rgb, img[..., 3:]], axis=2) if img.shape[2] == 4 else rgb


def texture(name):
    """Photo texture from assets/textures (see tools/fetch_assets.py), or None if missing."""
    path = os.path.join(ASSETS, f"{name}.jpg")
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("RGB"))


def _noise(size, scale, rng, octaves=4):
    """Tileable-ish fractal value noise in [0, 1]."""
    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = max(2, int(size / scale * 2**o))
        grid = rng.random((n, n)).astype(np.float32)
        img = Image.fromarray((grid * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        out += amp * np.asarray(img, np.float32) / 255.0
        total += amp
        amp *= 0.5
    return out / total


def leaves_texture(size=512, seed=0):
    """Dappled foliage: clumps of light and dark leaves."""
    rng = np.random.default_rng(seed)
    base = _noise(size, 64, rng)
    speck = rng.random((size, size)).astype(np.float32)
    speck = np.asarray(Image.fromarray((speck * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)),
                       np.float32) / 255.0
    v = np.clip(0.55 * base + 0.45 * speck, 0, 1)
    v = (v - v.min()) / (v.max() - v.min() + 1e-6)
    dark = np.array([22, 48, 18], np.float32)
    light = np.array([96, 140, 52], np.float32)
    rgb = dark + (light - dark) * (v[..., None] ** 1.6)
    return rgb.astype(np.uint8)


def plastic_texture(size=256, seed=1):
    """Near-white with faint mottling and vertical molding streaks; tint with the shape color."""
    rng = np.random.default_rng(seed)
    n = _noise(size, 32, rng, octaves=3)
    streak = np.tile(_noise(size, 8, rng, octaves=2)[:1], (size, 1))
    v = 0.9 + 0.06 * (n - 0.5) + 0.05 * (streak - 0.5)
    return (np.clip(v, 0, 1)[..., None].repeat(3, axis=2) * 255).astype(np.uint8)


def siding_texture(size=512):
    """Horizontal lap siding: light boards with a shadow line every ~20 cm (at 2 m per tile)."""
    y = np.arange(size)
    board = (y % (size // 10)) / (size // 10)
    v = 0.93 - 0.18 * np.exp(-((board - 0.0) ** 2) / 0.002) - 0.05 * board
    return (np.clip(v, 0, 1)[:, None, None].repeat(size, 1).repeat(3, 2) * 255).astype(np.uint8)


def shingles_texture(size=512, seed=2):
    """Asphalt shingle rows with per-tab variation."""
    rng = np.random.default_rng(seed)
    rows, cols = 16, 12
    v = np.zeros((size, size), np.float32)
    rh, cw = size // rows, size // cols
    for r in range(rows):
        off = (cw // 2) * (r % 2)
        for c in range(-1, cols + 1):
            x0 = c * cw + off
            v[r * rh:(r + 1) * rh, max(0, x0):max(0, min(size, x0 + cw - 2))] = rng.uniform(0.75, 1.0)
        v[r * rh:r * rh + 2] = 0.45
    v *= 0.92 + 0.08 * _noise(size, 16, rng, octaves=2)
    return (v[..., None].repeat(3, axis=2) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometry helpers


def _mesh(verts, faces, normals=None, uvs=None, tex=None):
    return newton.Mesh(
        np.asarray(verts, np.float32), np.asarray(faces, np.int32).reshape(-1),
        normals=None if normals is None else np.asarray(normals, np.float32),
        uvs=None if uvs is None else np.asarray(uvs, np.float32),
        texture=tex, compute_inertia=False,
    )


def _vertex_normals(verts, faces):
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces).reshape(-1, 3)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def ground_quad(x0, y0, x1, y1, tile, z=0.0, tex=None):
    verts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    uvs = [(x0 / tile, y0 / tile), (x1 / tile, y0 / tile), (x1 / tile, y1 / tile), (x0 / tile, y1 / tile)]
    return _mesh(verts, [0, 1, 2, 0, 2, 3], normals=[(0, 0, 1)] * 4, uvs=uvs, tex=tex)


def chaikin(pts, iterations=3):
    pts = [tuple(p) for p in pts]
    for _ in range(iterations):
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        out.append(pts[-1])
        pts = out
    return pts


def ribbon(pts, width, tile, z=0.0, tex=None, smooth=3, height_fn=None, step=0.5, columns=4):
    """Strip along a polyline (smoothed), UV-mapped along its length. Flat at `z`, or with `height_fn(x, y)`
    draped over the ground at `z` above it (resampled every `step` m, `columns` quads across)."""
    p = np.asarray(chaikin(pts, smooth) if smooth else pts, np.float64)
    if height_fn is not None:
        s_raw = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        si = np.linspace(0.0, s_raw[-1], max(2, int(s_raw[-1] / step) + 1))
        p = np.stack([np.interp(si, s_raw, p[:, 0]), np.interp(si, s_raw, p[:, 1])], axis=1)
    else:
        columns = 1
    seg = np.diff(p, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True) + 1e-12
    tang = np.vstack([seg[0], seg[:-1] + seg[1:], seg[-1]])
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    left = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    verts, uvs = [], []
    across = np.linspace(-1.0, 1.0, columns + 1)
    for i in range(len(p)):
        for side in across:
            q = p[i] + left[i] * side * width / 2
            verts.append((q[0], q[1], z))
            uvs.append(((side + 1) * width / 4 / tile, s[i] / tile))
    k = columns + 1
    faces = []
    for i in range(len(p) - 1):
        for j in range(columns):
            a, b, c, d = i * k + j, i * k + j + 1, (i + 1) * k + j, (i + 1) * k + j + 1
            faces += [a, c, b, b, c, d]
    if height_fn is None:
        return _mesh(verts, faces, normals=[(0, 0, 1)] * len(verts), uvs=uvs, tex=tex)
    v = np.asarray(verts, np.float64)
    v[:, 2] += height_fn(v[:, 0], v[:, 1])
    return _mesh(v, faces, normals=_vertex_normals(v, faces), uvs=uvs, tex=tex)


def ground_grid(x0, y0, x1, y1, step, tile, height_fn, tex=None):
    """Grass sheet draped over the terrain, UV-tiled every `tile` m."""
    xs = np.linspace(x0, x1, max(2, int((x1 - x0) / step) + 1))
    ys = np.linspace(y0, y1, max(2, int((y1 - y0) / step) + 1))
    gx, gy = np.meshgrid(xs, ys)
    v = np.stack([gx.ravel(), gy.ravel(), height_fn(gx.ravel(), gy.ravel())], axis=1)
    nx = len(xs)
    i = (np.arange(len(ys) - 1)[:, None] * nx + np.arange(nx - 1)[None, :]).ravel()
    faces = np.stack([i, i + 1, i + nx, i + 1, i + nx + 1, i + nx], axis=1).ravel()
    uvs = np.stack([v[:, 0] / tile, v[:, 1] / tile], axis=1)
    return _mesh(v, faces, normals=_vertex_normals(v, faces), uvs=uvs, tex=tex)


def tapered_cylinder(r0, r1, height, segments=16, tile=(1.0, 1.0), tex=None, jitter=0.0, rng=None):
    """Open-topped trunk along +Z with UVs wrapped around it."""
    rings = max(2, int(height / 0.5) + 1)
    verts, uvs = [], []
    for k in range(rings):
        t = k / (rings - 1)
        r = r0 + (r1 - r0) * t
        wob = 1.0 + (rng.uniform(-jitter, jitter) if rng is not None and jitter else 0.0)
        for s in range(segments + 1):
            a = 2 * math.pi * s / segments
            verts.append((r * wob * math.cos(a), r * wob * math.sin(a), height * t))
            uvs.append((s / segments * 2 * math.pi * r0 / tile[0], height * t / tile[1]))
    faces = []
    for k in range(rings - 1):
        for s in range(segments):
            a = k * (segments + 1) + s
            b, c, d = a + 1, a + segments + 1, a + segments + 2
            faces += [a, b, c, b, d, c]
    return _mesh(verts, faces, normals=_vertex_normals(verts, faces), uvs=uvs, tex=tex)


def _icosphere(subdiv):
    t = (1 + 5**0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    v = [np.array(x) / np.linalg.norm(x) for x in v]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f)


def canopy(radius, rng, lobes=5, tex=None):
    """A clumpy crown: a few overlapping, noise-displaced blobs merged into one mesh."""
    base_v, base_f = _icosphere(2)
    verts, faces = [], []
    for k in range(lobes):
        c = rng.normal(0, radius * 0.35, 3) * (1.0, 1.0, 0.6)
        r = radius * rng.uniform(0.55, 0.85)
        bumps = rng.normal(0, 1, (6, 3))
        n = base_v @ bumps.T
        disp = 1.0 + 0.12 * np.sin(n * 3.0).sum(axis=1) / 3.0
        v = base_v * (r * disp)[:, None] * (1.0, 1.0, 0.8) + c
        faces.append(base_f + len(verts) * 0 + sum(len(x) for x in verts))
        verts.append(v)
    verts = np.vstack(verts)
    faces = np.vstack(faces)
    return _mesh(verts, faces, normals=_vertex_normals(verts, faces), tex=tex)


def _rounded_rect(hx, hy, r, n=4):
    """Counter-clockwise outline of a rounded rectangle centered at the origin."""
    r = min(r, hx, hy)
    pts = []
    for cx, cy, a0 in ((hx - r, hy - r, 0), (-hx + r, hy - r, 90), (-hx + r, -hy + r, 180), (hx - r, -hy + r, 270)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def loft(sections, tex=None, tile=0.5, cap_bottom=True, cap_top=True):
    """Stack of closed outlines [(z, [(x, y), ...]), ...] with equal point counts -> closed mesh."""
    m = len(sections[0][1])
    verts, uvs = [], []
    for z, outline in sections:
        perim = 0.0
        prev = outline[0]
        for j, (x, y) in enumerate(outline + [outline[0]]):
            perim += math.dist(prev, (x, y))
            prev = (x, y)
            verts.append((x, y, z))
            uvs.append((perim / tile, z / tile))
    faces = []
    for k in range(len(sections) - 1):
        for j in range(m):
            a = k * (m + 1) + j
            b, c, d = a + 1, a + m + 1, a + m + 2
            faces += [a, b, c, b, d, c]
    for which, cap in ((0, cap_bottom), (len(sections) - 1, cap_top)):
        if not cap:
            continue
        z, outline = sections[which]
        cx = sum(p[0] for p in outline) / m
        cy = sum(p[1] for p in outline) / m
        center = len(verts)
        verts.append((cx, cy, z))
        uvs.append((0.0, 0.0))
        base = which * (m + 1)
        for j in range(m):
            a, b = base + j, base + j + 1
            faces += [center, b, a] if which == 0 else [center, a, b]
    # Flat-ish shading on caps and smooth on the sides comes out of averaged normals well enough here.
    return _mesh(verts, faces, normals=_vertex_normals(verts, faces), uvs=uvs, tex=tex)


def cart_body(depth, width, height, corner=0.07, slope=0.10, flare=0.04):
    """Wheelie-bin tub. +X is the back (vertical, where the wheels and handle are); the front
    (-X) slopes in toward the bottom and the sides flare out toward the top, like a real cart."""
    sections = []
    for t in np.linspace(0.0, 1.0, 9):
        hx_back = depth / 2
        hx_front = depth / 2 - slope * (1 - t)
        hy = width / 2 - flare * (1 - t)
        outline = _rounded_rect((hx_back + hx_front) / 2, hy, corner)
        shift = (hx_back - hx_front) / 2
        sections.append((height * t, [(x + shift, y) for x, y in outline]))
    return loft(sections, tile=0.4)


def cart_lid(depth, width, thickness=0.035, overhang=0.025, corner=0.07):
    outline = _rounded_rect(depth / 2 + overhang, width / 2 + overhang, corner + overhang)
    return loft([(0.0, outline), (thickness, outline)], tile=0.4)


def gable_roof(length, width, rise, overhang=0.4):
    """Roof prism over a footprint (length along X, width along Y), ridge along X."""
    L, W = length / 2 + overhang, width / 2 + overhang
    v = [(-L, -W, 0), (L, -W, 0), (L, 0, rise), (-L, 0, rise), (-L, W, 0), (L, W, 0)]
    f = [0, 1, 2, 0, 2, 3,  # south slope
         4, 3, 2, 4, 2, 5,  # north slope
         0, 3, 4, 1, 5, 2,  # gable ends
         0, 4, 5, 0, 5, 1]  # soffit
    # Duplicate vertices per face so slopes shade flat.
    verts, faces, uvs = [], [], []
    for i in range(0, len(f), 3):
        tri = [v[f[i]], v[f[i + 1]], v[f[i + 2]]]
        n = np.cross(np.subtract(tri[1], tri[0]), np.subtract(tri[2], tri[0]))
        horizontal = abs(n[2]) > 1e-6
        for p in tri:
            verts.append(p)
            # Slopes: U along the ridge, V up the slope; good enough for shingles.
            uvs.append((p[0] / 2.0, (math.hypot(p[1], p[2]) if horizontal else p[2]) / 2.0))
        faces += [len(verts) - 3, len(verts) - 2, len(verts) - 1]
    return _mesh(verts, faces, normals=_vertex_normals(verts, faces), uvs=uvs)
