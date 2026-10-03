import ARKit
import SceneKit
import SwiftUI

/// Builds an orbitable SceneKit scene of the captured mesh: ARKit-class colors, tagged objects in orange.
/// Uses ARKit world coordinates (Y-up, like SceneKit). Safe to call off the main thread.
enum ReviewScene {
    static func build(anchors: [ARMeshAnchor], start: SIMD3<Float>, boundsRadius: Float?, tags: [ObjectTag]) -> SCNScene {
        let raws = anchors.map { $0.worldMesh() }
        let split = MeshSplit(raws, start: start, boundsRadius: boundsRadius, tags: tags) { $0 }
        let scene = SCNScene()
        let parts = split.classParts.map(\.part) + tags.compactMap { split.objectParts[$0.id]?.part }
        for part in parts {
            if let node = node(for: part) { scene.rootNode.addChildNode(node) }
        }

        let marker = SCNNode(geometry: SCNSphere(radius: 0.08))
        marker.geometry?.firstMaterial?.diffuse.contents = UIColor.white
        marker.simdPosition = start
        scene.rootNode.addChildNode(marker)

        let (lo, hi) = scene.rootNode.boundingBox
        let center = SIMD3<Float>(Float(lo.x + hi.x) / 2, Float(lo.y + hi.y) / 2, Float(lo.z + hi.z) / 2)
        let size = max(simd_distance(SIMD3<Float>(Float(lo.x), Float(lo.y), Float(lo.z)),
                                     SIMD3<Float>(Float(hi.x), Float(hi.y), Float(hi.z))), 1)
        let camera = SCNCamera()
        camera.zNear = 0.01
        camera.zFar = Double(size) * 10
        let cameraNode = SCNNode()
        cameraNode.name = "reviewCamera"
        cameraNode.camera = camera
        cameraNode.simdPosition = center + SIMD3(0, size * 0.6, size * 0.6)
        cameraNode.simdLook(at: center)
        scene.rootNode.addChildNode(cameraNode)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 400
        scene.rootNode.addChildNode(ambient)
        let sun = SCNNode()
        sun.light = SCNLight()
        sun.light?.type = .directional
        sun.light?.intensity = 800
        sun.simdEulerAngles = SIMD3(-Float.pi / 3, Float.pi / 6, 0)
        scene.rootNode.addChildNode(sun)
        return scene
    }

    private static func node(for part: GLBWriter.Part) -> SCNNode? {
        guard !part.indices.isEmpty else { return nil }
        // Smooth vertex normals from accumulated face normals.
        var normals = [SIMD3<Float>](repeating: .zero, count: part.positions.count)
        for t in stride(from: 0, to: part.indices.count, by: 3) {
            let a = Int(part.indices[t]), b = Int(part.indices[t + 1]), c = Int(part.indices[t + 2])
            let n = simd_cross(part.positions[b] - part.positions[a], part.positions[c] - part.positions[a])
            normals[a] += n; normals[b] += n; normals[c] += n
        }
        var flatPositions = [Float]()
        var flatNormals = [Float]()
        flatPositions.reserveCapacity(part.positions.count * 3)
        flatNormals.reserveCapacity(part.positions.count * 3)
        for (p, n) in zip(part.positions, normals) {
            flatPositions.append(p.x); flatPositions.append(p.y); flatPositions.append(p.z)
            let len = simd_length(n)
            let u = len > 0 ? n / len : SIMD3<Float>(0, 1, 0)
            flatNormals.append(u.x); flatNormals.append(u.y); flatNormals.append(u.z)
        }

        func source(_ floats: [Float], _ semantic: SCNGeometrySource.Semantic) -> SCNGeometrySource {
            SCNGeometrySource(data: floats.withUnsafeBytes { Data($0) }, semantic: semantic,
                              vectorCount: floats.count / 3, usesFloatComponents: true,
                              componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        }
        let element = SCNGeometryElement(data: part.indices.withUnsafeBytes { Data($0) }, primitiveType: .triangles,
                                         primitiveCount: part.indices.count / 3, bytesPerIndex: 4)
        let geometry = SCNGeometry(sources: [source(flatPositions, .vertex), source(flatNormals, .normal)],
                                   elements: [element])
        let material = SCNMaterial()
        material.diffuse.contents = UIColor(red: CGFloat(part.color.x), green: CGFloat(part.color.y),
                                            blue: CGFloat(part.color.z), alpha: 1)
        material.lightingModel = .lambert
        material.isDoubleSided = true
        geometry.materials = [material]
        let node = SCNNode(geometry: geometry)
        node.name = part.name
        return node
    }
}

struct ReviewView: View {
    @ObservedObject var model: ScanModel

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let scene = model.reviewScene {
                OrbitView(scene: scene)
                    .ignoresSafeArea()
            } else {
                ProgressView("Building preview")
                    .tint(.white)
                    .foregroundStyle(.white)
            }
            VStack {
                HStack {
                    Text("Drag to orbit · pinch to zoom · two fingers to pan")
                        .font(.caption)
                        .padding(8)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                    Spacer()
                    Button("Done") { model.endReview() }
                        .buttonStyle(.borderedProminent)
                }
                Spacer()
                HStack(spacing: 12) {
                    legend("Floor", MeshClass.floor)
                    legend("Wall", 1)
                    legend("Other", 0)
                    HStack(spacing: 4) {
                        Circle().fill(Color.orange).frame(width: 8, height: 8)
                        Text("Tagged")
                    }
                }
                .font(.caption)
                .padding(8)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
            }
            .padding()
        }
    }

    private func legend(_ label: String, _ cls: UInt8) -> some View {
        let c = MeshClass.color(cls)
        return HStack(spacing: 4) {
            Circle().fill(Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))).frame(width: 8, height: 8)
            Text(label)
        }
    }
}

/// SCNView with turntable orbit, pinch zoom, and pan.
private struct OrbitView: UIViewRepresentable {
    let scene: SCNScene

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.backgroundColor = .black
        view.antialiasingMode = .multisampling4X
        view.allowsCameraControl = true
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.defaultCameraController.inertiaEnabled = true
        configure(view)
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        if view.scene !== scene { configure(view) }
    }

    private func configure(_ view: SCNView) {
        view.scene = scene
        view.pointOfView = scene.rootNode.childNode(withName: "reviewCamera", recursively: false)
        let (lo, hi) = scene.rootNode.boundingBox
        view.defaultCameraController.target = SCNVector3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
    }
}
