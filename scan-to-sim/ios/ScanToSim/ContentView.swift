import RealityKit
import SwiftUI

struct ContentView: View {
    @StateObject private var model = ScanModel()

    var body: some View {
        ZStack {
            ARViewContainer(model: model)
                .ignoresSafeArea()
            VStack(spacing: 12) {
                StatusPanel(model: model)
                if model.state == .scanning || model.state == .stopped {
                    HStack {
                        Spacer()
                        MiniMapView(map: model.miniMap)
                    }
                }
                Spacer()
                ControlPanel(model: model)
            }
            .padding()
            if model.state == .reviewing {
                ReviewView(model: model)
                    .transition(.opacity)
            }
        }
        .onAppear { model.prepare() }
        .sheet(item: $model.shareItem) { item in
            ShareSheet(items: [item.url])
        }
        .alert("Scan2Sim", isPresented: Binding(get: { model.errorMessage != nil },
                                                set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    let model: ScanModel

    func makeUIView(context: Context) -> ARView {
        let view = model.arView
        view.addGestureRecognizer(UITapGestureRecognizer(target: context.coordinator,
                                                         action: #selector(Coordinator.tapped(_:))))
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    final class Coordinator: NSObject {
        let model: ScanModel
        init(model: ScanModel) { self.model = model }

        @objc func tapped(_ gesture: UITapGestureRecognizer) {
            model.handleTap(at: gesture.location(in: gesture.view))
        }
    }
}

struct StatusPanel: View {
    @ObservedObject var model: ScanModel

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Circle()
                    .fill(model.trackingOK ? Color.green : Color.orange)
                    .frame(width: 10, height: 10)
                Text(model.trackingText)
                    .font(.subheadline.weight(.semibold))
            }
            if model.state != .idle {
                Text("\(model.keyframeCount) keyframes · \(model.faceCount.formatted()) faces · \(model.pathLength, specifier: "%.0f") m walked")
                    .font(.caption.monospacedDigit())
            }
            if model.state == .scanning && model.pathLength > ScanModel.driftWarningPath {
                Text("Long path: walk back over scanned ground to close a loop and limit drift.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            if let message = model.statusMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

struct ControlPanel: View {
    @ObservedObject var model: ScanModel
    private let boundsOptions: [Float] = [0, 10, 20, 30, 50]

    var body: some View {
        VStack(spacing: 10) {
            switch model.state {
            case .idle:
                TextField("Scan name", text: $model.scanName)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Bounds (radius around the start point)")
                        .font(.caption)
                    Picker("Bounds", selection: $model.boundsRadius) {
                        ForEach(boundsOptions, id: \.self) { r in
                            Text(r == 0 ? "Off" : "\(Int(r)) m").tag(r)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                Button {
                    model.start()
                } label: {
                    Label("Start scan", systemImage: "record.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

            case .scanning:
                Picker("Overlay", selection: $model.overlayMode) {
                    ForEach(ScanModel.OverlayMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                Toggle("Tag objects (tap to tag)", isOn: $model.tagMode)
                if model.tagMode {
                    HStack {
                        TextField("Object name", text: $model.tagName)
                            .textFieldStyle(.roundedBorder)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                        Stepper("\(model.tagRadius, specifier: "%.1f") m", value: $model.tagRadius, in: 0.2...3, step: 0.1)
                            .fixedSize()
                    }
                }
                HStack {
                    Button("Undo tag") { model.undoTag() }
                        .disabled(model.tags.isEmpty)
                    Text("\(model.tags.count) tagged")
                        .font(.caption)
                    Spacer()
                    Button {
                        model.stop()
                    } label: {
                        Label("Stop", systemImage: "stop.circle")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .controlSize(.large)
                }

            case .stopped:
                HStack {
                    Button(model.exported ? "New scan" : "Discard", role: model.exported ? nil : .destructive) {
                        model.reset()
                    }
                    Spacer()
                    Button {
                        model.review()
                    } label: {
                        Label("Review", systemImage: "rotate.3d")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    Button {
                        model.export()
                    } label: {
                        Label(model.exported ? "Export again" : "Export", systemImage: "square.and.arrow.up")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }

            case .reviewing:
                EmptyView()

            case .exporting:
                ProgressView(model.exportProgress)
                    .frame(maxWidth: .infinity)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
