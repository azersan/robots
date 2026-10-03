import ARKit
import Combine
import RealityKit
import SceneKit
import SwiftUI

struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}

/// Owns the AR session, the scan state, and the recorder. All members are used on the main thread
/// (ARSession delivers delegate calls on the main queue by default).
final class ScanModel: NSObject, ObservableObject, ARSessionDelegate {
    enum State { case idle, scanning, stopped, reviewing, exporting }

    enum OverlayMode: String, CaseIterable {
        case coverage = "Coverage"
        case wireframe = "Wireframe"
    }

    /// Path length after which ARKit drift becomes likely; the UI asks the user to close a loop.
    static let driftWarningPath: Float = 50

    @Published var state: State = .idle {
        didSet { applyOverlay() }
    }
    @Published var overlayMode: OverlayMode = .coverage {
        didSet { applyOverlay() }
    }
    @Published var scanName = "driveway"
    @Published var boundsRadius: Float = 30          // 0 = no bounds
    @Published var tagMode = false
    @Published var tagName = "barrel"
    @Published var tagRadius: Float = 0.6

    @Published private(set) var trackingText = "Starting camera"
    @Published private(set) var trackingOK = false
    @Published private(set) var keyframeCount = 0
    @Published private(set) var faceCount = 0
    @Published private(set) var pathLength: Float = 0
    @Published private(set) var tags: [ObjectTag] = []
    @Published private(set) var exportProgress = ""
    @Published var statusMessage: String?
    @Published var errorMessage: String?
    @Published var shareItem: ShareItem?
    @Published private(set) var exported = false
    @Published private(set) var reviewScene: SCNScene?

    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)

    private var meshAnchors: [UUID: ARMeshAnchor] = [:]
    private var recorder: KeyframeRecorder?
    private var startPose: simd_float4x4?
    private var lastPathPoint: SIMD3<Float>?
    private var startedAt = Date()
    private var tagEntities: [UUID: AnchorEntity] = [:]
    private var lastCoverageTick: TimeInterval = 0
    private var reviewBuildID = UUID()

    let coverage = CoverageOverlay()
    let miniMap = MiniMapState()

    override init() {
        super.init()
        arView.renderOptions.insert(.disableMotionBlur)
        arView.scene.addAnchor(coverage.root)
        arView.session.delegate = self
        applyOverlay()
    }

    /// Coverage overlay while a scan exists, otherwise ARKit's wireframe (also the fallback mode).
    private func applyOverlay() {
        let coverageOn = state != .idle && overlayMode == .coverage
        coverage.root.isEnabled = coverageOn
        if coverageOn {
            arView.debugOptions.remove(.showSceneUnderstanding)
        } else {
            arView.debugOptions.insert(.showSceneUnderstanding)
        }
    }

    var lidarSupported: Bool {
        ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification)
    }

    private func makeConfiguration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.sceneReconstruction = .meshWithClassification
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        return config
    }

    // MARK: - Lifecycle

    /// Runs the camera with the mesh overlay so the user can frame the scene before recording.
    func prepare() {
        guard lidarSupported else {
            errorMessage = "This device has no LiDAR. Scan2Sim needs an iPhone Pro or iPad Pro."
            return
        }
        arView.session.run(makeConfiguration(), options: [.resetTracking, .removeExistingAnchors])
    }

    func start() {
        guard lidarSupported else { prepare(); return }
        let stamp = Self.stampFormatter.string(from: Date())
        let dir = Self.scansDirectory.appendingPathComponent("\(Self.sanitize(scanName, fallback: "scan"))-\(stamp).scan")
        do {
            recorder = try KeyframeRecorder(bundleDir: dir)
        } catch {
            errorMessage = "Could not create scan folder: \(error.localizedDescription)"
            return
        }
        meshAnchors = [:]
        faceCount = 0
        keyframeCount = 0
        pathLength = 0
        lastPathPoint = nil
        startPose = nil
        startedAt = Date()
        clearTags()
        exported = false
        coverage.reset()
        miniMap.reset(boundsRadius: boundsRadius)
        reviewScene = nil
        reviewBuildID = UUID()
        arView.session.run(makeConfiguration(), options: [.resetTracking, .removeExistingAnchors])
        state = .scanning
    }

    func stop() {
        guard state == .scanning else { return }
        arView.session.pause()
        tagMode = false
        state = .stopped
    }

    /// Back to the idle camera. Deletes the scan folder unless it was exported.
    func reset() {
        if !exported, let dir = recorder?.bundleDir {
            try? FileManager.default.removeItem(at: dir)
        }
        recorder = nil
        meshAnchors = [:]
        faceCount = 0
        keyframeCount = 0
        pathLength = 0
        clearTags()
        coverage.reset()
        reviewScene = nil
        reviewBuildID = UUID()
        state = .idle
        prepare()
    }

    /// Orbit the captured mesh before export. The scene is built once per scan, off the main thread.
    func review() {
        guard state == .stopped else { return }
        state = .reviewing
        guard reviewScene == nil else { return }
        let buildID = UUID()
        reviewBuildID = buildID
        let anchors = Array(meshAnchors.values)
        let start = (startPose ?? matrix_identity_float4x4).translation
        let bounds: Float? = boundsRadius > 0 ? boundsRadius : nil
        let tags = tags
        DispatchQueue.global(qos: .userInitiated).async {
            let scene = ReviewScene.build(anchors: anchors, start: start, boundsRadius: bounds, tags: tags)
            DispatchQueue.main.async {
                guard self.reviewBuildID == buildID else { return }
                self.reviewScene = scene
            }
        }
    }

    func endReview() {
        guard state == .reviewing else { return }
        state = .stopped
    }

    func export() {
        guard state == .stopped, let recorder else { return }
        state = .exporting
        exportProgress = "Finishing keyframes"
        recorder.finish { [self] in
            let input = ExportInput(
                bundleDir: recorder.bundleDir,
                scanName: scanName,
                startedAt: startedAt,
                anchors: Array(meshAnchors.values),
                tags: tags,
                start: startPose ?? matrix_identity_float4x4,
                boundsRadius: boundsRadius > 0 ? boundsRadius : nil,
                keyframes: recorder.entries,
                pathLength: pathLength)
            DispatchQueue.global(qos: .userInitiated).async {
                let result = Result {
                    try Exporter.run(input) { step in
                        DispatchQueue.main.async { self.exportProgress = step }
                    }
                }
                DispatchQueue.main.async {
                    self.state = .stopped
                    switch result {
                    case .success(let zip):
                        self.exported = true
                        self.shareItem = ShareItem(url: zip)
                    case .failure(let error):
                        self.errorMessage = "Export failed: \(error.localizedDescription)"
                    }
                }
            }
        }
    }

    // MARK: - Object tagging

    func handleTap(at point: CGPoint) {
        guard state == .scanning, tagMode else { return }
        guard let hit = arView.raycast(from: point, allowing: .estimatedPlane, alignment: .any).first else {
            statusMessage = "No surface there yet. Scan it first."
            return
        }
        let center = hit.worldTransform.translation
        // A second tap on an already-tagged object would create an overlapping sliver object.
        if let existing = tags.first(where: { simd_distance($0.center, center) < $0.radius }) {
            statusMessage = "Already tagged as \(existing.name). Undo it first to re-tag."
            return
        }
        let tag = ObjectTag(id: UUID(), name: uniqueTagName(), center: center, radius: tagRadius)
        tags.append(tag)

        let anchor = AnchorEntity(world: center)
        let sphere = ModelEntity(mesh: .generateSphere(radius: tag.radius),
                                 materials: [SimpleMaterial(color: UIColor.systemOrange.withAlphaComponent(0.35),
                                                            isMetallic: false)])
        anchor.addChild(sphere)
        arView.scene.addAnchor(anchor)
        tagEntities[tag.id] = anchor
        miniMap.setTags(tags.map(\.center))
        statusMessage = "Tagged \(tag.name)"
    }

    func undoTag() {
        guard let last = tags.popLast() else { return }
        tagEntities.removeValue(forKey: last.id)?.removeFromParent()
        miniMap.setTags(tags.map(\.center))
    }

    private func clearTags() {
        tagEntities.values.forEach { $0.removeFromParent() }
        tagEntities = [:]
        tags = []
        miniMap.setTags([])
    }

    private func uniqueTagName() -> String {
        let base = Self.sanitize(tagName, fallback: "object")
        var name = base
        var n = 2
        while tags.contains(where: { $0.name == name }) {
            name = "\(base)_\(n)"
            n += 1
        }
        return name
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard state == .scanning else { return }
        let camera = frame.camera
        if startPose == nil, case .normal = camera.trackingState {
            startPose = camera.transform
            miniMap.setStart(camera.transform.translation)
        }
        miniMap.update(camera: camera.transform, time: frame.timestamp)

        let p = camera.transform.translation
        if let last = lastPathPoint {
            let d = simd_distance(p, last)
            if d > 0.1 {
                pathLength += d
                lastPathPoint = p
            }
        } else {
            lastPathPoint = p
        }

        if let recorder {
            recorder.consider(frame)
            if recorder.count != keyframeCount {
                keyframeCount = recorder.count
                if let k = recorder.entries.last { coverage.addKeyframe(k.transform) }
            }
        }
        if frame.timestamp - lastCoverageTick >= 0.5 {
            lastCoverageTick = frame.timestamp
            coverage.tick()
        }
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) { updateMeshes(anchors) }
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) { updateMeshes(anchors) }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        guard state == .scanning else { return }
        for anchor in anchors {
            meshAnchors.removeValue(forKey: anchor.identifier)
            coverage.remove(anchor.identifier)
        }
        recountFaces()
    }

    private func updateMeshes(_ anchors: [ARAnchor]) {
        guard state == .scanning else { return }
        var changed = false
        for case let mesh as ARMeshAnchor in anchors {
            meshAnchors[mesh.identifier] = mesh
            coverage.update(mesh)
            changed = true
        }
        if changed { recountFaces() }
    }

    private func recountFaces() {
        faceCount = meshAnchors.values.reduce(0) { $0 + $1.geometry.faces.count }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        switch camera.trackingState {
        case .normal:
            trackingText = "Tracking OK"
            trackingOK = true
        case .notAvailable:
            trackingText = "Tracking unavailable"
            trackingOK = false
        case .limited(let reason):
            trackingOK = false
            switch reason {
            case .excessiveMotion: trackingText = "Slow down"
            case .insufficientFeatures: trackingText = "Not enough detail. Aim at textured surfaces"
            case .initializing: trackingText = "Initializing. Move the phone slowly"
            case .relocalizing: trackingText = "Relocalizing. Return to a scanned area"
            @unknown default: trackingText = "Tracking limited"
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        errorMessage = "AR session failed: \(error.localizedDescription)"
    }

    // MARK: - Helpers

    static var scansDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("Scans", isDirectory: true)
    }

    static let stampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd-HHmm"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    /// Keeps names safe for file paths: letters, digits, '-' and '_'.
    static func sanitize(_ s: String, fallback: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let cleaned = String(s.trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: " ", with: "_")
            .unicodeScalars.filter { allowed.contains($0) })
        return cleaned.isEmpty ? fallback : cleaned
    }
}
