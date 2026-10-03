"""CLI: python -m scan2sim <bundle.scan | bundle.zip> [options]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compile import compile_bundle


def parse_masses(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--mass expects name=kg, got {item!r}")
        out[name] = float(value)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scan2sim", description="Compile a ScanToSim capture bundle into USD + MJCF.")
    ap.add_argument("bundle", type=Path, help=".scan folder or the .zip exported by the phone")
    ap.add_argument("--out", type=Path, help="output folder (default: <bundle>/sim)")
    ap.add_argument("--cell", type=float, default=0.05, help="heightfield cell size in meters (default 0.05)")
    ap.add_argument("--mass", action="append", default=[], metavar="NAME=KG", help="object mass, e.g. barrel=8")
    ap.add_argument("--default-mass", type=float, default=5.0, help="mass for objects without --mass (kg)")
    ap.add_argument("--friction", type=float, default=0.8)
    ap.add_argument("--collision", choices=["auto", "hull", "coacd"], default="auto",
                    help="object collision: hull to the ground, CoACD pieces, or auto (CoACD only if watertight)")
    ap.add_argument("--no-texture", action="store_true", help="skip coloring from keyframes")
    ap.add_argument("--min-confidence", type=int, default=1, choices=[0, 1, 2], help="lowest LiDAR depth confidence used")
    ap.add_argument("--corridor", type=float, metavar="M",
                    help="keep only geometry within M meters of the walked path (tagged objects are always kept)")
    ap.add_argument("--max-faces", type=int, help="decimate the scene mesh to this many faces")
    ap.add_argument("--check", action="store_true", help="after compiling, drop a box in MuJoCo and verify it rests")
    ap.add_argument("--newton", action="store_true", help="with --check, also run the drop test in Newton (scene.usda)")
    args = ap.parse_args(argv)

    compile_bundle(args.bundle, out=args.out, cell=args.cell, masses=parse_masses(args.mass),
                   default_mass=args.default_mass, friction=args.friction, collision=args.collision,
                   texture=not args.no_texture, min_confidence=args.min_confidence, max_faces=args.max_faces,
                   corridor=args.corridor)
    if args.check:
        from .bundle import open_bundle
        from .check import mujoco_check

        sim = args.out or open_bundle(args.bundle) / "sim"
        results = [mujoco_check(sim)]
        if args.newton:
            from .check import newton_check

            results.append(newton_check(sim))
        for result in results:
            print(json.dumps(result, indent=2))
        return 0 if all(r["pass"] for r in results) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
