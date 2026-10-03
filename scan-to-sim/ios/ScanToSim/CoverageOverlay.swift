import ARKit
import RealityKit

/// Semi-transparent mesh overlay colored by how many keyframes have seen each face
/// (red = none, green = 5 or more). Driven from the main thread; counting runs on a background queue.
final class CoverageOverlay {
    /// A keyframe counts for a face when the face is within this distance of the camera...
    static let maxDistance: Float = 4
    /// ...and in front of it: cos(angle between view direction and the ray to the face).
    static let minDot: Float = 0.5

    /// Level colors: 0, 1, 2, 3-4, 5+ keyframes.
    static let levelColors: [UIColor] = [
        UIColor(red: 0.90, green: 0.15, blue: 0.10, alpha: 1),
        UIColor(red: 0.95, green: 0.50, blue: 0.10, alpha: 1),
        UIColor(red: 0.95, green: 0.85, blue: 0.15, alpha: 1),
        UIColor(red: 0.60, green: 0.85, blue: 0.20, alpha: 1),
        UIColor(red: 0.15, green: 0.80, blue: 0.30, alpha: 1),
    ]

    let root = AnchorEntity(world: matrix_identity_float4x4)

    private let materials: [RealityKit.Material]
    private var entities: [UUID: ModelEntity] = [:]

    // Pending work, collected on the main thread between ticks.
    private var dirtyAnchors: [UUID: ARMeshAnchor] = [:]
    private var removedAnchors: Set<UUID> = []
    private var newCameras: [CoverageWorker.Camera] = []
    private var busy = false
    private var generation = 0

    private let queue = DispatchQueue(label: "coverage", qos: .userInitiated)
    private let worker = CoverageWorker()   // touched only on `queue`

    init() {
        materials = Self.levelColors.map { color in
            var m = UnlitMaterial(color: color)
            m.blending = .transparent(opacity: 0.45)
            return m
        }
    }

    func update(_ anchor: ARMeshAnchor) {
        dirtyAnchors[anchor.identifier] = anchor
        removedAnchors.remove(anchor.identifier)
    }

    func remove(_ id: UUID) {
        dirtyAnchors.removeValue(forKey: id)
        removedAnchors.insert(id)
    }

    func addKeyframe(_ transform: simd_float4x4) {
        let c = transform.columns.2
        newCameras.append(.init(position: transform.translation, forward: -simd_normalize(SIMD3(c.x, c.y, c.z))))
    }

    /// Sends pending changes to the worker unless a batch is still in flight. Call at ~2 Hz.
    func tick() {
        guard !busy, !(dirtyAnchors.isEmpty && removedAnchors.isEmpty && newCameras.isEmpty) else { return }
        busy = true
        let gen = generation
        let updated = Array(dirtyAnchors.values)
        let removed = removedAnchors
        let cameras = newCameras
        dirtyAnchors = [:]
        removedAnchors = []
        newCameras = []
        queue.async { [worker] in
            let render = worker.process(updated: updated, removed: removed, newCameras: cameras)
            DispatchQueue.main.async {
                self.busy = false
                guard gen == self.generation else { return }
                self.apply(render, removed: removed)
            }
        }
    }

    func reset() {
        generation += 1
        entities.values.forEach { $0.removeFromParent() }
        entities = [:]
        dirtyAnchors = [:]
        removedAnchors = []
        newCameras = []
        queue.async { [worker] in worker.reset() }
    }

    private func apply(_ render: [CoverageWorker.RenderData], removed: Set<UUID>) {
        for id in removed {
            entities.removeValue(forKey: id)?.removeFromParent()
        }
        for r in render {
            var descriptor = MeshDescriptor(name: "coverage")
            descriptor.positions = MeshBuffers.Positions(r.positions)
            descriptor.primitives = .triangles(r.indices)
            descriptor.materials = .perFace(r.faceLevels)
            guard let mesh = try? MeshResource.generate(from: [descriptor]) else { continue }
            if let entity = entities[r.id] {
                entity.model?.mesh = mesh
            } else {
                let entity = ModelEntity(mesh: mesh, materials: materials)
                root.addChild(entity)
                entities[r.id] = entity
            }
        }
    }
}

/// Background state for CoverageOverlay: per-anchor face centroids and view counts.
final class CoverageWorker {
    struct Camera {
        let position: SIMD3<Float>
        let forward: SIMD3<Float>
    }

    struct RenderData {
        let id: UUID
        let positions: [SIMD3<Float>]
        let indices: [UInt32]
        let faceLevels: [UInt32]
    }

    private struct AnchorData {
        var positions: [SIMD3<Float>]
        var indices: [UInt32]
        var centroids: [SIMD3<Float>]
        var center: SIMD3<Float>
        var radius: Float
        var counts: [UInt8]
        var levels: [UInt8]
    }

    private var anchors: [UUID: AnchorData] = [:]
    private var cameras: [Camera] = []

    func reset() {
        anchors = [:]
        cameras = []
    }

    func process(updated: [ARMeshAnchor], removed: Set<UUID>, newCameras: [Camera]) -> [RenderData] {
        for id in removed { anchors.removeValue(forKey: id) }
        let updatedIDs = Set(updated.map(\.identifier))
        var changed = Set<UUID>()

        // New keyframes against anchors whose geometry did not change.
        for cam in newCameras {
            for id in Array(anchors.keys) where !updatedIDs.contains(id) {
                if Self.apply(cam, to: &anchors[id]!) { changed.insert(id) }
            }
        }
        cameras += newCameras

        // Changed geometry: recount against every keyframe so far.
        for anchor in updated {
            var data = Self.makeData(anchor.worldMesh())
            for cam in cameras { _ = Self.apply(cam, to: &data) }
            anchors[anchor.identifier] = data
            changed.insert(anchor.identifier)
        }

        return changed.compactMap { id in
            guard let a = anchors[id], !a.indices.isEmpty else { return nil }
            return RenderData(id: id, positions: a.positions, indices: a.indices, faceLevels: a.levels.map(UInt32.init))
        }
    }

    private static func makeData(_ raw: RawAnchorMesh) -> AnchorData {
        var indices = [UInt32]()
        indices.reserveCapacity(raw.faces.count * 3)
        var centroids = [SIMD3<Float>]()
        centroids.reserveCapacity(raw.faces.count)
        for f in raw.faces {
            indices.append(f.x); indices.append(f.y); indices.append(f.z)
            centroids.append((raw.vertices[Int(f.x)] + raw.vertices[Int(f.y)] + raw.vertices[Int(f.z)]) / 3)
        }
        var center = SIMD3<Float>.zero
        for v in raw.vertices { center += v }
        center /= Float(max(raw.vertices.count, 1))
        let radius = raw.vertices.reduce(Float(0)) { max($0, simd_distance($1, center)) }
        return AnchorData(positions: raw.vertices, indices: indices, centroids: centroids, center: center,
                          radius: radius, counts: Array(repeating: 0, count: centroids.count),
                          levels: Array(repeating: 0, count: centroids.count))
    }

    /// Adds one keyframe's view to an anchor's faces. Returns true if any face changed color.
    private static func apply(_ cam: Camera, to a: inout AnchorData) -> Bool {
        guard simd_distance(cam.position, a.center) < CoverageOverlay.maxDistance + a.radius else { return false }
        var changed = false
        for i in a.centroids.indices {
            let d = a.centroids[i] - cam.position
            let dist = simd_length(d)
            guard dist > 0.05, dist < CoverageOverlay.maxDistance,
                  simd_dot(d / dist, cam.forward) > CoverageOverlay.minDot else { continue }
            if a.counts[i] < 255 { a.counts[i] += 1 }
            let level = Self.level(a.counts[i])
            if level != a.levels[i] {
                a.levels[i] = level
                changed = true
            }
        }
        return changed
    }

    private static func level(_ count: UInt8) -> UInt8 {
        switch count {
        case 0: return 0
        case 1: return 1
        case 2: return 2
        case 3, 4: return 3
        default: return 4
        }
    }
}
