import Foundation
import simd

/// Minimal binary glTF 2.0 writer: one node + mesh + material per part, positions and uint32 indices only.
/// Coordinates are written as given (the bundle uses the Z-up sim frame, not glTF's Y-up; see manifest.json).
enum GLBWriter {
    struct Part {
        let name: String
        let positions: [SIMD3<Float>]
        let indices: [UInt32]
        let color: SIMD4<Float>
    }

    static func write(_ parts: [Part], to url: URL) throws {
        var bin = Data()
        var bufferViews: [[String: Any]] = []
        var accessors: [[String: Any]] = []
        var materials: [[String: Any]] = []
        var meshes: [[String: Any]] = []
        var nodes: [[String: Any]] = []

        for part in parts where !part.indices.isEmpty {
            var flat = [Float]()
            flat.reserveCapacity(part.positions.count * 3)
            var lo = SIMD3<Float>(repeating: .infinity)
            var hi = SIMD3<Float>(repeating: -.infinity)
            for v in part.positions {
                flat.append(v.x); flat.append(v.y); flat.append(v.z)
                lo = simd_min(lo, v)
                hi = simd_max(hi, v)
            }

            let posOffset = bin.count
            flat.withUnsafeBytes { bin.append(contentsOf: $0) }
            bufferViews.append(["buffer": 0, "byteOffset": posOffset, "byteLength": flat.count * 4, "target": 34962])
            accessors.append(["bufferView": bufferViews.count - 1, "componentType": 5126,
                              "count": part.positions.count, "type": "VEC3",
                              "min": [lo.x, lo.y, lo.z].map { Double($0) },
                              "max": [hi.x, hi.y, hi.z].map { Double($0) }])
            let posAccessor = accessors.count - 1

            let idxOffset = bin.count
            part.indices.withUnsafeBytes { bin.append(contentsOf: $0) }
            bufferViews.append(["buffer": 0, "byteOffset": idxOffset, "byteLength": part.indices.count * 4, "target": 34963])
            accessors.append(["bufferView": bufferViews.count - 1, "componentType": 5125,
                              "count": part.indices.count, "type": "SCALAR"])
            let idxAccessor = accessors.count - 1

            materials.append(["name": part.name, "doubleSided": true,
                              "pbrMetallicRoughness": ["baseColorFactor": [part.color.x, part.color.y, part.color.z, part.color.w].map { Double($0) },
                                                       "metallicFactor": 0.0, "roughnessFactor": 1.0]])
            meshes.append(["name": part.name,
                           "primitives": [["attributes": ["POSITION": posAccessor], "indices": idxAccessor,
                                           "material": materials.count - 1, "mode": 4]]])
            nodes.append(["name": part.name, "mesh": meshes.count - 1])
        }

        var gltf: [String: Any] = [
            "asset": ["version": "2.0", "generator": "ScanToSim"],
            "scene": 0,
            "scenes": [["nodes": Array(0..<nodes.count)]],
        ]
        if !nodes.isEmpty {
            gltf["nodes"] = nodes
            gltf["meshes"] = meshes
            gltf["materials"] = materials
            gltf["accessors"] = accessors
            gltf["bufferViews"] = bufferViews
            gltf["buffers"] = [["byteLength": bin.count]]
        }

        var json = try JSONSerialization.data(withJSONObject: gltf)
        while json.count % 4 != 0 { json.append(0x20) }
        while bin.count % 4 != 0 { bin.append(0) }

        var out = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        let total = 12 + 8 + json.count + (bin.isEmpty ? 0 : 8 + bin.count)
        u32(0x4654_6C67)            // "glTF"
        u32(2)
        u32(UInt32(total))
        u32(UInt32(json.count))
        u32(0x4E4F_534A)            // "JSON"
        out.append(json)
        if !bin.isEmpty {
            u32(UInt32(bin.count))
            u32(0x004E_4942)        // "BIN\0"
            out.append(bin)
        }
        try out.write(to: url)
    }
}
