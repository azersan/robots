# Scan-to-Sim

iPhone Pro (LiDAR) app that scans an area and exports a capture bundle for the
robotics sim. Spec: "Scan-to-Sim iPhone App — Spec" (Claude Docs).

## Build & run

```bash
cd ios
xcodegen generate          # regenerate ScanToSim.xcodeproj after editing project.yml or adding files
open ScanToSim.xcodeproj   # pick your iPhone Pro as the run destination, then Run
```

Requires a LiDAR device (iPhone 12 Pro or newer Pro, or iPad Pro). Signing uses team `JC3W536XWY`, automatic.

## Using it

1. Name the scan, pick bounds (radius around where you start), tap **Start scan**.
2. Walk the area slowly. The mesh is drawn over the camera, colored by coverage: red = no
   keyframe has seen it yet, orange/yellow = 1–4, green = 5+ (a keyframe counts if it was
   within 4 m and facing the surface). Switch the overlay to **Wireframe** if you prefer
   ARKit's raw mesh. The mini-map (top right) shows your path, the bounds circle, the start
   point (white), tags (orange) and you (yellow arrow); map-up is the direction you faced at
   start (the sim's +Y). Outdoors, walk in loops; after 50 m of path the app reminds you to
   close a loop.
3. To make an object its own rigid body: turn on **Tag objects**, set its name and radius,
   tap it. Faces inside the orange sphere (except ARKit "floor") go to `objects/<name>.glb`.
4. **Stop**. **Review** lets you orbit the captured mesh (drag to orbit, pinch to zoom, two
   fingers to pan; class colors, tagged objects in orange, bounds applied). Then **Export**. A share sheet offers the zip (AirDrop, Files…). Bundles also
   stay in the app's Documents/Scans folder, visible in the Files app.

## Bundle format (`<name>-<date>.scan/`)

| File | Contents |
| --- | --- |
| `manifest.json` | version, device, units, frame, origin, bounds, parts, objects |
| `mesh.glb` | merged untextured mesh, one node per ARKit class (hint only) |
| `objects/<name>.glb` | one per tagged object |
| `keyframes/NNNNN.heic` / `.depth.f32` / `.conf.u8` | image, float32 depth (m), uint8 confidence |
| `poses.json` | per keyframe: camera-to-world (row-major), intrinsics, sizes |
| `preview.obj` | Y-up preview (iOS can't write USDZ via ModelIO, so this falls back to OBJ) |

Frame: meters, Z-up, right-handed; sim = (x, −z, y) of ARKit world, origin at the start
point projected to the ground. The `.glb` files store sim-frame coordinates directly
(not glTF's usual Y-up), so the PC-side `scan2sim` compiler reads them as-is.

## Scanning tips (learned on the driveway)

- Start at the end you want as the sim origin (the garage for Anna Pl), facing down the path. Bounds **Off**
  for long paths; use `--corridor` in scan2sim instead.
- Walk out slowly with the phone low and tilted down, then walk back the same way.
- Finish where you started, **facing walls, posts or edges you saw at the start**: the drift report needs
  upright structure in common (smooth ground alone can't pin down drift, and it then says "unreliable").
- Tag each object once (a second tap inside a tag's sphere is refused).
- A full out-and-back of ~90 m exports roughly 400 MB; AirDrop it to `~/Documents/Meshes`.

## From a scan to the sim playground

```bash
# 1. Compile the bundle (Mac or PC): ground heightfield, collision, USD + MJCF, drift report.
cd scan-to-sim/scan2sim
python -m scan2sim ~/Documents/Meshes/<name>.scan.zip --corridor 3 --ground depth --check

# 2. Turn it into sim-playground terrain for the Anna Pl site (writes sim-playground/terrain/anna_pl.npz).
cd ../../sim-playground
python tools/import_scan.py ~/Documents/Meshes/<name>.scan

# 3. Commit terrain/anna_pl.npz; on the NZXT run the scene `anna_pl_scan` (see sim-playground/README.md).
```

- `--ground depth` matters for out-and-back walks: drift between the two passes otherwise leaves phantom
  10–15 cm steps in the ground that stop the car. Details and every option: [`scan2sim/README.md`](scan2sim/README.md).
- `import_scan.py` places the scan by fitting the walked path onto the site's traced route and fills the
  unscanned part of the site smoothly (marked as unmeasured).

Status (2026-10-03): the first driveway scan covers ~45 m from the garage (half the route). In the playground,
`tow_home.py` fetched and towed a bin home on that ground in 2/2 trials. Next: scan the full ~90 m route,
finishing at the start facing the garage, and re-import.
