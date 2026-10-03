import ARKit
import ModelIO
import UIKit

struct ObjectTag: Identifiable {
    let id: UUID
    let name: String
    let center: SIMD3<Float>   // ARKit world
    let radius: Float
}

struct ExportInput {
    let bundleDir: URL
    let scanName: String
    let startedAt: Date
    let anchors: [ARMeshAnchor]
    let tags: [ObjectTag]
    let start: simd_float4x4        // camera pose when tracking first became normal
    let boundsRadius: Float?        // horizontal circle around the start point
    let keyframes: [KeyframeEntry]
    let pathLength: Float
}

/// Writes the capture bundle described in the spec and zips it. Runs off the main thread.
enum Exporter {
    static func run(_ input: ExportInput, progress: @escaping (String) -> Void) throws -> URL {
        let fm = FileManager.default
        let dir = input.bundleDir

        progress("Reading mesh")
        let raws = input.anchors.map { $0.worldMesh() }
        let start = input.start.translation
        let groundY = estimateGroundY(raws, near: start)
        let frame = SimFrame(origin: SIMD3(start.x, groundY ?? start.y - 1.4, start.z))

        progress("Splitting mesh")
        let split = MeshSplit(raws, start: start, boundsRadius: input.boundsRadius, tags: input.tags,
                              transform: frame.point)
        let objectParts = split.objectParts
        let droppedOutside = split.droppedOutside

        progress("Writing meshes")
        let meshParts = split.classParts
        try GLBWriter.write(meshParts.map(\.part), to: dir.appendingPathComponent("mesh.glb"))

        var objectsJSON: [[String: Any]] = []
        let tagsWithFaces = input.tags.filter { objectParts[$0.id] != nil }
        if !tagsWithFaces.isEmpty {
            try fm.createDirectory(at: dir.appendingPathComponent("objects"), withIntermediateDirectories: true)
        }
        for tag in tagsWithFaces {
            guard let part = objectParts[tag.id] else { continue }
            let file = "objects/\(tag.name).glb"
            try GLBWriter.write([part.part], to: dir.appendingPathComponent(file))
            let c = frame.point(tag.center)
            objectsJSON.append(["name": tag.name, "file": file, "faces": part.faceCount,
                                "tag_center": [c.x, c.y, c.z].map(round6), "tag_radius_m": round6(tag.radius)])
        }

        progress("Writing poses")
        try writePoses(input.keyframes, frame: frame, to: dir.appendingPathComponent("poses.json"))

        progress("Writing preview")
        let preview = writePreview(meshParts.map(\.part) + tagsWithFaces.compactMap { objectParts[$0.id]?.part }, in: dir)

        progress("Writing manifest")
        let manifest: [String: Any] = [
            "format": "scan2sim-bundle",
            "version": 1,
            "name": input.scanName,
            "created": ISO8601DateFormatter().string(from: input.startedAt),
            "device": deviceModel(),
            "app_version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?",
            "units": "meters",
            "frame": "z_up_right_handed",
            "frame_note": "All geometry and poses use the sim frame: sim = (x, -z, y) of ARKit world, minus the origin. "
                + "The .glb files hold sim-frame coordinates directly (they are NOT glTF Y-up). preview.* is Y-up for Quick Look.",
            "origin": [
                "description": "scan start camera position projected to the ground",
                "arkit_world": [frame.origin.x, frame.origin.y, frame.origin.z].map(round6),
                "ground_estimated": groundY != nil,
            ],
            "bounds": input.boundsRadius.map { ["type": "circle", "radius_m": round6($0)] as [String: Any] } ?? NSNull(),
            "faces_dropped_outside_bounds": droppedOutside,
            "path_length_m": round6(input.pathLength),
            "mesh": [
                "file": "mesh.glb",
                "note": "one node per ARKit classification; classes are hints only, ground/background split happens in scan2sim",
                "parts": meshParts.map { ["name": $0.name, "faces": $0.faceCount] },
            ],
            "objects": objectsJSON,
            "keyframes": ["count": input.keyframes.count, "poses": "poses.json", "dir": "keyframes/"],
            "preview": preview ?? NSNull(),
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))

        progress("Zipping")
        return try zip(dir)
    }

    /// Ground height (ARKit Y) near the start point: 10th percentile of surface heights
    /// within 1.5 m horizontally and at least 0.3 m below the camera. Nil if too few points.
    static func estimateGroundY(_ raws: [RawAnchorMesh], near p: SIMD3<Float>) -> Float? {
        var ys: [Float] = []
        for raw in raws {
            for v in raw.vertices {
                let dx = v.x - p.x, dz = v.z - p.z
                if dx * dx + dz * dz < 2.25, v.y < p.y - 0.3 { ys.append(v.y) }
            }
        }
        guard ys.count >= 50 else { return nil }
        ys.sort()
        return ys[ys.count / 10]
    }

    static func writePoses(_ keyframes: [KeyframeEntry], frame: SimFrame, to url: URL) throws {
        let list: [[String: Any]] = keyframes.map { k in
            var e: [String: Any] = [
                "index": k.index,
                "timestamp": k.timestamp,
                "image": k.image,
                "depth": k.depth,
                "image_size": [k.imageSize.width, k.imageSize.height],
                "depth_size": [k.depthSize.width, k.depthSize.height],
                "intrinsics": k.intrinsics.rowMajor,
                "camera_to_world": frame.pose(k.transform).rowMajor,
            ]
            if let c = k.confidence { e["confidence"] = c }
            return e
        }
        let doc: [String: Any] = [
            "camera_convention": "ARKit/OpenGL camera axes: +x right, +y up, camera looks along -z",
            "camera_to_world": "row-major 4x4, sim frame (Z-up, meters)",
            "intrinsics": "row-major 3x3 in pixels at image_size; scale by depth_size/image_size for depth maps",
            "depth_format": "float32 little-endian meters, row-major, depth_size",
            "confidence_format": "uint8 per pixel, 0=low 1=medium 2=high, depth_size",
            "image_note": "HEIC in landscape sensor orientation, as captured",
            "keyframes": list,
        ]
        let data = try JSONSerialization.data(withJSONObject: doc, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }

    /// Quick Look preview in Y-up (USDZ if ModelIO can write it, else OBJ). Returns the file name.
    static func writePreview(_ parts: [GLBWriter.Part], in dir: URL) -> String? {
        let asset = MDLAsset()
        let allocator = MDLMeshBufferDataAllocator()
        let descriptor = MDLVertexDescriptor()
        descriptor.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition, format: .float3, offset: 0, bufferIndex: 0)
        descriptor.layouts[0] = MDLVertexBufferLayout(stride: 12)

        for p in parts where !p.indices.isEmpty {
            var flat = [Float]()
            flat.reserveCapacity(p.positions.count * 3)
            for v in p.positions { flat.append(v.x); flat.append(v.z); flat.append(-v.y) }  // sim Z-up -> Y-up
            let vbuf = allocator.newBuffer(with: flat.withUnsafeBytes { Data($0) }, type: .vertex)
            let ibuf = allocator.newBuffer(with: p.indices.withUnsafeBytes { Data($0) }, type: .index)
            let material = MDLMaterial(name: p.name, scatteringFunction: MDLPhysicallyPlausibleScatteringFunction())
            material.setProperty(MDLMaterialProperty(name: "baseColor", semantic: .baseColor,
                                                     float3: SIMD3(p.color.x, p.color.y, p.color.z)))
            let submesh = MDLSubmesh(indexBuffer: ibuf, indexCount: p.indices.count, indexType: .uInt32,
                                     geometryType: .triangles, material: material)
            let mesh = MDLMesh(vertexBuffer: vbuf, vertexCount: p.positions.count,
                               descriptor: descriptor, submeshes: [submesh])
            mesh.name = p.name
            asset.add(mesh)
        }
        guard asset.count > 0 else { return nil }

        for ext in ["usdz", "obj"] where MDLAsset.canExportFileExtension(ext) {
            let url = dir.appendingPathComponent("preview.\(ext)")
            if (try? asset.export(to: url)) != nil { return url.lastPathComponent }
        }
        return nil
    }

    /// Zips a directory next to itself using NSFileCoordinator's .forUploading option.
    static func zip(_ dir: URL) throws -> URL {
        let dest = dir.deletingPathExtension().appendingPathExtension("scan.zip")
        var coordError: NSError?
        var copyError: Error?
        NSFileCoordinator().coordinate(readingItemAt: dir, options: .forUploading, error: &coordError) { tmp in
            do {
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.copyItem(at: tmp, to: dest)
            } catch {
                copyError = error
            }
        }
        if let e = coordError ?? copyError { throw e }
        return dest
    }

    static func deviceModel() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }
}
