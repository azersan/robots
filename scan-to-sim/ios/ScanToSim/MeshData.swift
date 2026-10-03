import ARKit

/// One ARKit mesh anchor, copied out of its Metal buffers into world space.
struct RawAnchorMesh {
    var vertices: [SIMD3<Float>]   // ARKit world coordinates
    var faces: [SIMD3<UInt32>]
    var classes: [UInt8]           // ARMeshClassification raw value per face
}

enum MeshClass {
    static let names = ["none", "wall", "floor", "ceiling", "table", "seat", "window", "door"]
    static let floor: UInt8 = 2

    static func name(_ raw: UInt8) -> String { Int(raw) < names.count ? names[Int(raw)] : "class\(raw)" }

    static func color(_ raw: UInt8) -> SIMD4<Float> {
        switch raw {
        case 1: return SIMD4(0.70, 0.70, 0.75, 1)   // wall
        case 2: return SIMD4(0.45, 0.65, 0.40, 1)   // floor
        case 3: return SIMD4(0.85, 0.85, 0.90, 1)   // ceiling
        case 4, 5: return SIMD4(0.75, 0.55, 0.35, 1) // table, seat
        case 6, 7: return SIMD4(0.40, 0.55, 0.80, 1) // window, door
        default: return SIMD4(0.60, 0.60, 0.60, 1)  // none
        }
    }
}

extension ARMeshAnchor {
    func worldMesh() -> RawAnchorMesh {
        let g = geometry

        let vs = g.vertices
        let vbase = vs.buffer.contents().advanced(by: vs.offset)
        var vertices = [SIMD3<Float>]()
        vertices.reserveCapacity(vs.count)
        for i in 0..<vs.count {
            let p = vbase.advanced(by: i * vs.stride).assumingMemoryBound(to: Float.self)
            let w = transform * SIMD4<Float>(p[0], p[1], p[2], 1)
            vertices.append(SIMD3(w.x, w.y, w.z))
        }

        let f = g.faces
        let fbase = f.buffer.contents()
        var faces = [SIMD3<UInt32>]()
        faces.reserveCapacity(f.count)
        for i in 0..<f.count {
            var tri = SIMD3<UInt32>()
            for k in 0..<3 {
                let off = (i * f.indexCountPerPrimitive + k) * f.bytesPerIndex
                tri[k] = f.bytesPerIndex == 4
                    ? fbase.load(fromByteOffset: off, as: UInt32.self)
                    : UInt32(fbase.load(fromByteOffset: off, as: UInt16.self))
            }
            faces.append(tri)
        }

        var classes = [UInt8](repeating: 0, count: f.count)
        if let c = g.classification {
            let cbase = c.buffer.contents().advanced(by: c.offset)
            for i in 0..<min(c.count, f.count) {
                classes[i] = cbase.load(fromByteOffset: i * c.stride, as: UInt8.self)
            }
        }
        return RawAnchorMesh(vertices: vertices, faces: faces, classes: classes)
    }
}

/// Accumulates faces into one indexed mesh, re-indexing vertices per source anchor.
final class PartBuilder {
    let name: String
    let color: SIMD4<Float>
    private(set) var positions: [SIMD3<Float>] = []
    private(set) var indices: [UInt32] = []
    private var remap: [UInt32: UInt32] = [:]

    init(name: String, color: SIMD4<Float>) {
        self.name = name
        self.color = color
    }

    /// Call before adding faces from a new anchor (vertex indices restart per anchor).
    func beginAnchor() { remap.removeAll(keepingCapacity: true) }

    func addFace(_ face: SIMD3<UInt32>, vertices: [SIMD3<Float>]) {
        for k in 0..<3 {
            let v = face[k]
            if let m = remap[v] {
                indices.append(m)
            } else {
                let n = UInt32(positions.count)
                positions.append(vertices[Int(v)])
                remap[v] = n
                indices.append(n)
            }
        }
    }

    var faceCount: Int { indices.count / 3 }

    var part: GLBWriter.Part { GLBWriter.Part(name: name, positions: positions, indices: indices, color: color) }
}

/// Splits anchor meshes into per-class parts and per-tag object parts, dropping faces outside the bounds.
/// Shared by export (sim frame) and review (ARKit world).
struct MeshSplit {
    /// Sorted by ARKit class raw value.
    let classParts: [PartBuilder]
    let objectParts: [UUID: PartBuilder]
    let droppedOutside: Int

    static let objectColor = SIMD4<Float>(0.95, 0.55, 0.15, 1)

    /// - Parameters:
    ///   - start: scan start in ARKit world; the bounds circle is centered on it.
    ///   - transform: maps ARKit world points into the output frame.
    init(_ raws: [RawAnchorMesh], start: SIMD3<Float>, boundsRadius: Float?, tags: [ObjectTag],
         transform: (SIMD3<Float>) -> SIMD3<Float>) {
        var classParts: [UInt8: PartBuilder] = [:]
        var objectParts: [UUID: PartBuilder] = [:]
        var dropped = 0
        for raw in raws {
            let outVerts = raw.vertices.map(transform)
            classParts.values.forEach { $0.beginAnchor() }
            objectParts.values.forEach { $0.beginAnchor() }
            for (i, face) in raw.faces.enumerated() {
                let centroid = (raw.vertices[Int(face.x)] + raw.vertices[Int(face.y)] + raw.vertices[Int(face.z)]) / 3
                if let r = boundsRadius {
                    let dx = centroid.x - start.x, dz = centroid.z - start.z
                    if dx * dx + dz * dz > r * r { dropped += 1; continue }
                }
                let cls = raw.classes[i]
                if cls != MeshClass.floor,
                   let tag = tags.first(where: { simd_distance($0.center, centroid) <= $0.radius }) {
                    let part = objectParts[tag.id] ?? PartBuilder(name: tag.name, color: Self.objectColor)
                    objectParts[tag.id] = part
                    part.addFace(face, vertices: outVerts)
                } else {
                    let part = classParts[cls] ?? PartBuilder(name: MeshClass.name(cls), color: MeshClass.color(cls))
                    classParts[cls] = part
                    part.addFace(face, vertices: outVerts)
                }
            }
        }
        self.classParts = classParts.keys.sorted().compactMap { classParts[$0] }
        self.objectParts = objectParts
        self.droppedOutside = dropped
    }
}
