"""OpenUSD scene for Newton (add_usd) and Isaac Sim: Z-up, meters, UsdPhysics colliders and rigid bodies."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def _mesh(stage, path: str, v: np.ndarray, f: np.ndarray, colors: np.ndarray | None = None,
          rgb: tuple | None = None) -> UsdGeom.Mesh:
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(f.ravel(), dtype=np.int32)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)
    lo, hi = v.min(axis=0), v.max(axis=0)
    m.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    if colors is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(colors, dtype=np.float32)))
    elif rgb is not None:
        m.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    return m


def _collider(mesh: UsdGeom.Mesh, approximation: str):
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collision-only geometry is not rendered


def _physics_material(stage, path: str, friction: float, restitution: float = 0.0) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(restitution)
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def write(path: Path, ground_visual, ground_collision, background, objects: list[dict], friction: float = 0.8):
    """objects: dicts with name, origin (3,), mass, visual (Part in sim frame), pieces [(v, f)] in sim frame."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    ground_mat = _physics_material(stage, "/World/Materials/Ground", friction)
    object_mat = _physics_material(stage, "/World/Materials/Object", friction)

    UsdGeom.Xform.Define(stage, "/World/Ground")
    if ground_visual is not None and len(ground_visual.faces):
        _mesh(stage, "/World/Ground/Visual", ground_visual.vertices, ground_visual.faces, ground_visual.colors, (0.45, 0.6, 0.4))
    gv, gf = ground_collision
    col = _mesh(stage, "/World/Ground/Collision", gv, gf)
    _collider(col, "none")
    _bind(col.GetPrim(), ground_mat)

    if background is not None and len(background.faces):
        UsdGeom.Xform.Define(stage, "/World/Background")
        _mesh(stage, "/World/Background/Visual", background.vertices, background.faces, background.colors, (0.6, 0.6, 0.6))
        bcol = _mesh(stage, "/World/Background/Collision", background.vertices, background.faces)
        _collider(bcol, "none")
        _bind(bcol.GetPrim(), ground_mat)

    UsdGeom.Scope.Define(stage, "/World/Objects")
    for obj in objects:
        root = f"/World/Objects/{obj['name']}"
        xf = UsdGeom.Xform.Define(stage, root)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, obj["origin"])))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(obj["mass"]))
        _bind(prim, object_mat)
        vis = obj["visual"]
        _mesh(stage, f"{root}/Visual", vis.vertices - obj["origin"], vis.faces, vis.colors, (0.95, 0.55, 0.15))
        for i, (v, f) in enumerate(obj["pieces"]):
            c = _mesh(stage, f"{root}/Collision_{i}", v - obj["origin"], f)
            _collider(c, "convexHull")

    stage.GetRootLayer().Save()
