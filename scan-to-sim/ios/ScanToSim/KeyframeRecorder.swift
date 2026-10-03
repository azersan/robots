import ARKit
import CoreImage

struct KeyframeEntry {
    let index: Int
    let timestamp: TimeInterval
    let transform: simd_float4x4      // camera-to-world, ARKit world
    let intrinsics: simd_float3x3     // at imageSize
    let imageSize: (width: Int, height: Int)
    let depthSize: (width: Int, height: Int)
    let image: String                 // paths relative to the bundle root
    let depth: String
    let confidence: String?
}

/// Saves a keyframe (HEIC image, raw depth, confidence) every ~0.25 m or 10 degrees of camera motion.
/// Must be driven from the main thread; encoding and file writes happen on a background queue.
final class KeyframeRecorder {
    static let minTranslation: Float = 0.25
    static let minRotation: Float = 10 * .pi / 180
    /// Frames being encoded at once. Each one holds an ARKit camera buffer, so keep this small.
    static let maxInFlight = 2

    let bundleDir: URL
    private(set) var entries: [KeyframeEntry] = []
    private var lastPose: simd_float4x4?
    private var inFlight = 0
    private let queue = DispatchQueue(label: "keyframe-writer", qos: .utility)
    private let ciContext = CIContext()

    init(bundleDir: URL) throws {
        self.bundleDir = bundleDir
        try FileManager.default.createDirectory(at: bundleDir.appendingPathComponent("keyframes"),
                                                withIntermediateDirectories: true)
    }

    var count: Int { entries.count }

    func consider(_ frame: ARFrame) {
        guard case .normal = frame.camera.trackingState, inFlight < Self.maxInFlight,
              let depth = frame.sceneDepth else { return }
        let pose = frame.camera.transform
        if let last = lastPose,
           simd_distance(pose.translation, last.translation) < Self.minTranslation,
           rotationAngle(last, pose) < Self.minRotation {
            return
        }
        lastPose = pose

        let index = entries.count
        let stem = String(format: "keyframes/%05d", index)
        let depthData = Self.copyPlane(depth.depthMap)
        let confidenceData = depth.confidenceMap.map(Self.copyPlane)
        let image = frame.capturedImage
        let entry = KeyframeEntry(
            index: index, timestamp: frame.timestamp, transform: pose, intrinsics: frame.camera.intrinsics,
            imageSize: (CVPixelBufferGetWidth(image), CVPixelBufferGetHeight(image)),
            depthSize: (CVPixelBufferGetWidth(depth.depthMap), CVPixelBufferGetHeight(depth.depthMap)),
            image: stem + ".heic", depth: stem + ".depth.f32",
            confidence: confidenceData == nil ? nil : stem + ".conf.u8")
        entries.append(entry)

        inFlight += 1
        let dir = bundleDir
        queue.async { [ciContext] in
            let ci = CIImage(cvPixelBuffer: image)
            let options = [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.85]
            if let heic = ciContext.heifRepresentation(of: ci, format: .RGBA8,
                                                       colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                                       options: options) {
                try? heic.write(to: dir.appendingPathComponent(entry.image))
            }
            try? depthData.write(to: dir.appendingPathComponent(entry.depth))
            if let confidenceData, let name = entry.confidence {
                try? confidenceData.write(to: dir.appendingPathComponent(name))
            }
            DispatchQueue.main.async { self.inFlight -= 1 }
        }
    }

    /// Calls `completion` on the main thread once every queued keyframe is on disk.
    func finish(_ completion: @escaping () -> Void) {
        queue.async { DispatchQueue.main.async(execute: completion) }
    }

    /// Copies a single-plane pixel buffer into tightly packed rows.
    private static func copyPlane(_ buffer: CVPixelBuffer) -> Data {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        // Depth is DepthFloat32; confidence is OneComponent8.
        let pixelBytes = CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_DepthFloat32 ? 4 : 1
        let packedRow = width * pixelBytes
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return Data() }
        var data = Data(capacity: packedRow * height)
        for row in 0..<height {
            data.append(base.advanced(by: row * rowBytes).assumingMemoryBound(to: UInt8.self), count: packedRow)
        }
        return data
    }
}
