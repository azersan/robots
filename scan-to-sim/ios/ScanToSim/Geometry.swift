import simd

extension simd_float4x4 {
    var translation: SIMD3<Float> { SIMD3(columns.3.x, columns.3.y, columns.3.z) }

    var rotation3x3: simd_float3x3 {
        simd_float3x3(SIMD3(columns.0.x, columns.0.y, columns.0.z),
                      SIMD3(columns.1.x, columns.1.y, columns.1.z),
                      SIMD3(columns.2.x, columns.2.y, columns.2.z))
    }

    /// Row-major flattening for JSON (simd matrices are column-major).
    var rowMajor: [Double] { (0..<4).flatMap { r in (0..<4).map { c in round6(self[c][r]) } } }
}

extension simd_float3x3 {
    var rowMajor: [Double] { (0..<3).flatMap { r in (0..<3).map { c in round6(self[c][r]) } } }
}

func round6(_ x: Float) -> Double { (Double(x) * 1e6).rounded() / 1e6 }

/// Angle in radians between the orientations of two poses.
func rotationAngle(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
    let q = simd_quatf(a.rotation3x3).inverse * simd_quatf(b.rotation3x3)
    let angle = q.angle
    return min(angle, 2 * .pi - angle)
}

/// Converts ARKit world coordinates (Y-up) into the sim frame:
/// Z-up, right-handed, meters, origin at the scan start point projected to the ground.
struct SimFrame {
    /// Sim origin expressed in ARKit world coordinates.
    let origin: SIMD3<Float>

    /// (x, y, z) ARKit -> (x, -z, y) sim. A proper rotation (det = +1).
    static let rotation = simd_float3x3(rows: [SIMD3(1, 0, 0), SIMD3(0, 0, -1), SIMD3(0, 1, 0)])

    func point(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let d = p - origin
        return SIMD3(d.x, -d.z, d.y)
    }

    /// Camera-to-world pose in ARKit world -> camera-to-world pose in sim frame.
    func pose(_ t: simd_float4x4) -> simd_float4x4 {
        let r = SimFrame.rotation * t.rotation3x3
        let p = point(t.translation)
        return simd_float4x4(columns: (SIMD4(r.columns.0, 0), SIMD4(r.columns.1, 0),
                                       SIMD4(r.columns.2, 0), SIMD4(p, 1)))
    }
}
