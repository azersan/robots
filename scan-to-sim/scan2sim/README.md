# scan2sim

Compiles a ScanToSim capture bundle (the `.scan.zip` the phone exports) into sim scenes:

```bash
pip install -r requirements.txt
python -m scan2sim ~/Documents/Meshes/test_1-2026-10-03-1652.scan.zip --check
python -m scan2sim driveway.scan.zip --mass barrel=8 --check --newton
python -m scan2sim path.scan.zip --corridor 3 --check     # long path: keep 3 m either side of where you walked
```

Every run prints the **loop drift**, e.g. `Loop drift: 27 cm over 180 m walked (0.15%), yaw 1.5 deg: usable`.
It is measured only if you finish where you started, facing the way you set off: the depth seen then is aligned
(ICP) with the depth seen at the start, and the offset is how far the camera poses drifted over the walk.
Under 20 cm is good; 20–50 cm means some ghosting; over 50 cm, split the path into overlapping scans.
Keyframe poses are never corrected after the fact, so high drift also smears colors from the far end.

Output goes to `<bundle>.scan/sim/`:

| File | What |
| --- | --- |
| `scene.usda` | Newton / Isaac Sim scene: Z-up, meters, UsdPhysics. Ground collider = heightfield mesh, background = static triangle mesh, tagged objects = rigid bodies with convex colliders. Visuals carry per-vertex colors from the keyframes. |
| `scene.xml` | MuJoCo fallback: heightfield ground (`ground.hfield.bin`), free-jointed objects. Background is visual-only here. |
| `collision/*.obj` | Convex collision pieces per object (sim frame). |
| `meshes/*.obj` | Meshes referenced by the MJCF. |
| `report.json` | Heightfield extent and coverage, face counts, objects, texture coverage. |

## Pipeline

1. Corridor (optional): drop faces more than `--corridor` meters from the walked path (keyframe positions).
   Tagged objects are always kept.
2. Clean the mesh: weld vertices, drop degenerate faces and floating fragments under 30 faces (`--max-faces` decimates).
3. Ground: per 5 cm cell, the lowest **up-facing** surface (ARKit winds faces consistently). Cells more than
   0.3 m above the local floor level (tabletops, seats) are rejected, then holes are filled from neighbors.
   ARKit classes are not used.
4. Split: faces on the heightfield are ground, the rest is background.
5. Objects: ground faces are removed from each tagged region, then only the connected piece at the tap point is kept
   (neighbors caught by the tag sphere go back to the background). Tags with < 100 faces are skipped (double taps).
6. Color: each vertex takes the color of the closest keyframe that sees it, checked against that keyframe's LiDAR depth,
   ignoring depth below `--min-confidence` (default 1 = medium).
7. Collision: hull of the object's points plus their projection to the ground (scans are open shells); CoACD pieces
   with `--collision coacd` or when the object mesh is watertight.

## Checks

`--check` drops a 20 cm box at the scan origin in MuJoCo and verifies it rests on the ground, and that tagged objects
stay put. `--newton` runs the same test through `newton.ModelBuilder.add_usd`.

Known issue (Newton 1.6, XPBD solver, CPU): the box rests correctly, but rigid bodies that *start* in contact with the
triangle-mesh ground get launched. Test on the PC with Newton's MuJoCo solver (`SolverMuJoCo`, needs a matching
`mujoco-warp`) before tuning anything.

Synthetic end-to-end test (sloped driveway, barrel, wall, ray-traced keyframes; plus a walked path
out and back with planted drift to test the drift report and corridor):

```bash
python tests/test_pipeline.py
```
