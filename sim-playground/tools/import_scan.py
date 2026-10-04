"""Turn a phone LiDAR scan (scan2sim output) into terrain for a site: terrain/<site>.npz.

    python tools/import_scan.py ~/scans/driveway-2026-10-03-1746.scan [--site anna_pl] [--cell 0.1]

The scan is placed in the site frame by fitting the path walked during the scan (keyframe positions) onto
the site's hand-traced route: a rigid 2D fit (rotation + shift; the scan is metric), tried from both ends
of the route and a spread of headings. Its heightfield is resampled onto an axis-aligned grid over the
site bounds. Cells the scan didn't measure are filled smoothly from the nearest scanned ground, so the
unscanned part of the route is a plausible guess, not data (`measured` in the npz says which is which).

Expects <scan>/poses.json and <scan>/sim/{report.json, ground.hfield.bin, ground.mask.bin} from
`python -m scan2sim <zip> --corridor 3 --ground depth` (see scan-to-sim/scan2sim).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simpg import site_anna_pl  # noqa: E402

SITES = {"anna_pl": site_anna_pl}


def read_grid(path):
    raw = np.fromfile(path, dtype=np.uint8)
    ny, nx = raw[:8].view("<i4")
    return ny, nx, raw[8:]


def load_scan(scan: Path):
    rep = json.loads((scan / "sim" / "report.json").read_text())["heightfield"]
    ny, nx, data = read_grid(scan / "sim" / rep["file"])
    z = data.view("<f4").reshape(ny, nx).astype(np.float64)
    mask_file = scan / "sim" / rep.get("mask", "ground.mask.bin")
    if mask_file.exists():
        _, _, m = read_grid(mask_file)
        measured = m.reshape(ny, nx).astype(bool)
    else:
        measured = np.ones_like(z, dtype=bool)
    poses = json.loads((scan / "poses.json").read_text())["keyframes"]
    path = np.array([np.array(k["camera_to_world"]).reshape(4, 4)[:2, 3] for k in poses])
    return rep, z, measured, path


def outbound(path):
    """The walk away from the start, minus any looking around at the start."""
    d = np.linalg.norm(path - path[0], axis=1)
    far = int(np.argmax(d))
    begin = int(np.nonzero(d[: far + 1] < 1.0)[0].max())
    return path[begin : far + 1]


def densify(pts, step=0.2):
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n, endpoint=False)]
    return np.array(out + [pts[-1]])


def rot(th):
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def fit_rigid(walk, route):
    """2D rotation + translation mapping scan xy onto the route polyline (ICP, multi-start)."""
    dense = densify(route)
    tree = cKDTree(dense)
    best = None
    wdir = math.atan2(*(walk[min(10, len(walk) - 1)] - walk[0])[::-1])
    for anchor, nxt in [(route[0], route[1]), (route[-1], route[-2])]:
        base = math.atan2(*(nxt - anchor)[::-1])
        for dyaw in np.radians(np.arange(-60, 61, 10)):
            th = base - wdir + dyaw
            t = anchor - rot(th) @ walk[0]
            for _ in range(80):
                x = walk @ rot(th).T + t
                _, j = tree.query(x)
                q = dense[j]
                mp, mq = walk.mean(0), q.mean(0)
                H = (walk - mp).T @ (q - mq)
                th = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
                t = mq - rot(th) @ mp
            dist, _ = tree.query(walk @ rot(th).T + t)
            score = float(np.median(dist))
            if best is None or score < best[0]:
                best = (score, float(np.percentile(dist, 90)), th, t)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", type=Path, help="<name>.scan folder (scan2sim output)")
    ap.add_argument("--site", default="anna_pl", choices=list(SITES))
    ap.add_argument("--cell", type=float, default=0.1, help="terrain grid cell (m)")
    ap.add_argument("--margin", type=float, default=15.0, help="grid extends this far past the route (m)")
    ap.add_argument("--out", type=Path, help="default: terrain/<site>.npz")
    args = ap.parse_args()
    site = SITES[args.site]

    rep, z_scan, meas_scan, path = load_scan(args.scan)
    walk = outbound(path)
    route = np.array(site.route_m())
    med, p90, th, t = fit_rigid(walk, route)
    start_site = rot(th) @ walk[0] + t
    print(f"scan placed: rotation {math.degrees(th):.1f} deg, scan start at site ({start_site[0]:.1f}, "
          f"{start_site[1]:.1f}); walked path vs route: median {med:.2f} m, 90% within {p90:.2f} m")

    # Site grid covering the route plus a margin.
    x0, y0 = route.min(0) - args.margin
    x1, y1 = route.max(0) + args.margin
    nx, ny = int(math.ceil((x1 - x0) / args.cell)) + 1, int(math.ceil((y1 - y0) / args.cell)) + 1
    gx, gy = np.meshgrid(x0 + np.arange(nx) * args.cell, y0 + np.arange(ny) * args.cell)
    # Site xy -> scan xy (inverse of the fit), then to scan grid indices.
    pts = np.stack([gx.ravel(), gy.ravel()], 1) - t
    sxy = pts @ rot(th)  # rot(th).T inverse applied row-wise
    fx = (sxy[:, 0] - rep["x0"]) / rep["cell"]
    fy = (sxy[:, 1] - rep["y0"]) / rep["cell"]
    sny, snx = z_scan.shape
    inside = (fx >= 0) & (fx <= snx - 1) & (fy >= 0) & (fy <= sny - 1)
    z = np.full(gx.size, np.nan)
    z[inside] = ndimage.map_coordinates(z_scan, [fy[inside], fx[inside]], order=1)
    near = np.zeros(gx.size, bool)
    near[inside] = ndimage.map_coordinates(meas_scan.astype(float), [fy[inside], fx[inside]], order=0) > 0.5
    z = np.where(near, z, np.nan).reshape(ny, nx)
    measured = near.reshape(ny, nx)
    if not measured.any():
        raise SystemExit("scan does not overlap the site grid")

    # Fill the unscanned ground: nearest scanned height, then smoothed so it reads as gentle terrain.
    idx = ndimage.distance_transform_edt(~measured, return_distances=False, return_indices=True)
    filled = z[tuple(idx)]
    smooth = ndimage.gaussian_filter(filled, sigma=3.0 / args.cell)
    # Blend: exact inside the scan, easing to the smooth fill over ~2 m outside it.
    dist = ndimage.distance_transform_edt(~measured) * args.cell
    w = np.clip(dist / 2.0, 0.0, 1.0)
    z = np.where(measured, z, (1 - w) * filled + w * smooth).astype(np.float32)

    out = args.out or Path(__file__).resolve().parents[1] / "terrain" / f"{args.site}.npz"
    out.parent.mkdir(exist_ok=True)
    meta = {"source": args.scan.name, "site": args.site, "rotation_deg": round(math.degrees(th), 3),
            "translation": [round(float(v), 4) for v in t], "fit_median_m": round(med, 3),
            "fit_p90_m": round(p90, 3), "scan_start_site": [round(float(v), 2) for v in start_site],
            "measured_fraction": round(float(measured.mean()), 4)}
    np.savez_compressed(out, x0=np.float64(x0), y0=np.float64(y0), cell=np.float64(args.cell), z=z,
                        measured=measured, meta=json.dumps(meta))
    print(f"wrote {out}: {nx}x{ny} cells at {args.cell} m, {measured.mean() * 100:.0f}% scanned, "
          f"z {z.min():.2f}..{z.max():.2f} m ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
