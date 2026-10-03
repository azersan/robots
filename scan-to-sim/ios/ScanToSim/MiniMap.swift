import SwiftUI
import simd

/// Top-down map state, kept separate from ScanModel so its ~5 Hz updates only redraw the map.
/// Map coordinates: (x, -z) of ARKit world, so up on the map is the direction the phone faced
/// when the scan started (the sim frame's +Y).
final class MiniMapState: ObservableObject {
    static let pathSpacing: Float = 0.5
    static let maxPathPoints = 2000
    static let publishInterval: TimeInterval = 0.2

    @Published private(set) var path: [SIMD2<Float>] = []
    @Published private(set) var position: SIMD2<Float>?
    @Published private(set) var heading: Float = 0      // radians clockwise from map-up
    @Published private(set) var start: SIMD2<Float>?
    @Published private(set) var boundsRadius: Float = 0
    @Published private(set) var tags: [SIMD2<Float>] = []

    private var lastPublish: TimeInterval = 0

    static func mapPoint(_ p: SIMD3<Float>) -> SIMD2<Float> { SIMD2(p.x, -p.z) }

    func reset(boundsRadius: Float) {
        path = []
        position = nil
        start = nil
        tags = []
        heading = 0
        lastPublish = 0
        self.boundsRadius = boundsRadius
    }

    func setStart(_ p: SIMD3<Float>) { start = Self.mapPoint(p) }

    func setTags(_ centers: [SIMD3<Float>]) { tags = centers.map(Self.mapPoint) }

    func update(camera transform: simd_float4x4, time: TimeInterval) {
        guard time - lastPublish >= Self.publishInterval else { return }
        lastPublish = time
        let p = Self.mapPoint(transform.translation)
        let f = transform.columns.2   // camera looks along -z
        heading = atan2(-f.x, f.z)    // map forward = (-f.x, f.z)
        position = p
        if let last = path.last, simd_distance(last, p) < Self.pathSpacing { return }
        path.append(p)
        if path.count > Self.maxPathPoints {
            path = path.enumerated().filter { $0.offset % 2 == 0 }.map(\.element)
        }
    }
}

struct MiniMapView: View {
    @ObservedObject var map: MiniMapState

    var body: some View {
        Canvas { ctx, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let usable = Float(min(size.width, size.height) / 2 - 8)
            let origin = map.start ?? map.position ?? .zero

            var extent = max(3, map.boundsRadius)
            for p in map.path { extent = max(extent, simd_distance(p, origin) + 1) }
            if let p = map.position { extent = max(extent, simd_distance(p, origin) + 1) }
            let scale = CGFloat(usable / extent)

            func pt(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(x: center.x + CGFloat(p.x - origin.x) * scale, y: center.y - CGFloat(p.y - origin.y) * scale)
            }

            if map.boundsRadius > 0 {
                let r = CGFloat(map.boundsRadius) * scale
                ctx.stroke(Path(ellipseIn: CGRect(x: center.x - r, y: center.y - r, width: 2 * r, height: 2 * r)),
                           with: .color(.white.opacity(0.6)), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
            }

            if map.path.count > 1 {
                var line = Path()
                line.addLines(map.path.map(pt))
                ctx.stroke(line, with: .color(.cyan), lineWidth: 2)
            }

            if map.start != nil {
                ctx.fill(Path(ellipseIn: CGRect(x: center.x - 4, y: center.y - 4, width: 8, height: 8)), with: .color(.white))
            }

            for t in map.tags {
                let c = pt(t)
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - 4, y: c.y - 4, width: 8, height: 8)), with: .color(.orange))
            }

            if let p = map.position {
                let c = pt(p)
                var arrow = Path()
                arrow.move(to: CGPoint(x: 0, y: -8))
                arrow.addLine(to: CGPoint(x: 5, y: 6))
                arrow.addLine(to: CGPoint(x: 0, y: 3))
                arrow.addLine(to: CGPoint(x: -5, y: 6))
                arrow.closeSubpath()
                let t = CGAffineTransform(translationX: c.x, y: c.y).rotated(by: CGFloat(map.heading))
                ctx.fill(arrow.applying(t), with: .color(.yellow))
            }
        }
        .frame(width: 130, height: 130)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}
