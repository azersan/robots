"""Loading a capture bundle written by the ScanToSim iOS app (see scan-to-sim/README.md)."""
from __future__ import annotations

import json
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Part:
    name: str
    vertices: np.ndarray  # (N, 3) float64, sim frame (Z-up, meters)
    faces: np.ndarray  # (M, 3) int64
    colors: np.ndarray | None = None  # (N, 3) float32 in 0..1, filled by texturing
    meta: dict = field(default_factory=dict)


@dataclass
class Keyframe:
    index: int
    image: Path
    depth: Path
    confidence: Path | None
    image_size: tuple[int, int]  # (width, height)
    depth_size: tuple[int, int]
    K: np.ndarray  # 3x3 intrinsics at image_size
    cam_to_world: np.ndarray  # 4x4, sim frame, OpenGL camera axes (+x right, +y up, looks along -z)

    def load_depth(self) -> np.ndarray:
        w, h = self.depth_size
        return np.fromfile(self.depth, dtype="<f4").reshape(h, w)

    def load_confidence(self) -> np.ndarray | None:
        if self.confidence is None or not self.confidence.exists():
            return None
        w, h = self.depth_size
        return np.fromfile(self.confidence, dtype=np.uint8).reshape(h, w)


@dataclass
class Bundle:
    root: Path
    manifest: dict
    mesh_parts: list[Part]
    objects: list[Part]
    keyframes: list[Keyframe]


def open_bundle(path: Path) -> Path:
    """Returns the bundle directory, extracting a .zip next to itself if needed."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        if (path / "manifest.json").exists():
            return path
        raise FileNotFoundError(f"no manifest.json in {path}")
    if path.suffix != ".zip":
        raise ValueError(f"expected a .scan folder or .zip, got {path}")
    dest = path.parent / path.name[: -len(".zip")]
    if not dest.name.endswith(".scan"):
        dest = dest.with_name(dest.name + ".scan")
    if not (dest / "manifest.json").exists():
        with zipfile.ZipFile(path) as z:
            z.extractall(dest.parent / (dest.name + ".tmp"))
        tmp = dest.parent / (dest.name + ".tmp")
        manifests = list(tmp.rglob("manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"no manifest.json inside {path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        manifests[0].parent.rename(dest)
        _rmtree(tmp)
    return dest


def _rmtree(p: Path):
    import shutil

    shutil.rmtree(p, ignore_errors=True)


def read_glb(path: Path) -> list[Part]:
    """Reads the minimal GLB the app writes: one node/mesh/primitive per part, positions + uint32 indices."""
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"{path} is not a glTF 2.0 binary")
    json_len, _ = struct.unpack_from("<II", data, 12)
    doc = json.loads(data[20 : 20 + json_len])
    off = 20 + json_len
    binary = b""
    if off < len(data):
        bin_len, _ = struct.unpack_from("<II", data, off)
        binary = data[off + 8 : off + 8 + bin_len]

    def accessor(i: int, dtype, width: int) -> np.ndarray:
        acc = doc["accessors"][i]
        view = doc["bufferViews"][acc["bufferView"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        arr = np.frombuffer(binary, dtype=dtype, count=acc["count"] * width, offset=start)
        return arr.reshape(-1, width) if width > 1 else arr

    parts = []
    for node in doc.get("nodes", []):
        mesh = doc["meshes"][node["mesh"]]
        for prim in mesh["primitives"]:
            v = accessor(prim["attributes"]["POSITION"], "<f4", 3).astype(np.float64)
            idx = accessor(prim["indices"], "<u4", 1).astype(np.int64).reshape(-1, 3)
            parts.append(Part(name=node.get("name", mesh.get("name", "part")), vertices=v, faces=idx))
    return parts


def load(path: Path) -> Bundle:
    root = open_bundle(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "scan2sim-bundle":
        raise ValueError(f"{root} is not a scan2sim bundle")
    mesh_parts = read_glb(root / manifest["mesh"]["file"])

    objects = []
    for obj in manifest.get("objects", []):
        parts = read_glb(root / obj["file"])
        if not parts:
            continue
        objects.append(_merge(parts, obj["name"], meta=obj))

    keyframes = []
    poses_file = root / manifest.get("keyframes", {}).get("poses", "poses.json")
    if poses_file.exists():
        poses = json.loads(poses_file.read_text())
        for k in poses["keyframes"]:
            image = root / k["image"]
            if not image.exists():
                continue
            keyframes.append(
                Keyframe(
                    index=k["index"],
                    image=image,
                    depth=root / k["depth"],
                    confidence=root / k["confidence"] if k.get("confidence") else None,
                    image_size=tuple(k["image_size"]),
                    depth_size=tuple(k["depth_size"]),
                    K=np.array(k["intrinsics"], dtype=np.float64).reshape(3, 3),
                    cam_to_world=np.array(k["camera_to_world"], dtype=np.float64).reshape(4, 4),
                )
            )
    return Bundle(root=root, manifest=manifest, mesh_parts=mesh_parts, objects=objects, keyframes=keyframes)


def _merge(parts: list[Part], name: str, meta: dict | None = None) -> Part:
    verts, faces, n = [], [], 0
    for p in parts:
        verts.append(p.vertices)
        faces.append(p.faces + n)
        n += len(p.vertices)
    return Part(name=name, vertices=np.concatenate(verts), faces=np.concatenate(faces), meta=meta or {})


merge_parts = _merge
