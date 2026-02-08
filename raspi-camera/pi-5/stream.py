#!/usr/bin/env python3
"""
MJPEG streaming server for Raspberry Pi 5 with Camera 3.

Features:
  - 640x480 @ 30fps default (configurable up to 1080p)
  - Continuous autofocus
  - High quality JPEG encoding
  - FPS monitoring

Usage:
    python3 stream.py                        # Defaults
    python3 stream.py -r 1280x720            # 720p
    python3 stream.py -r 1920x1080 -f 15     # 1080p at 15fps
"""

from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import argparse
import time

app = Flask(__name__)
camera = None
jpeg_quality = 85
frame_count = 0
fps_start_time = None


def parse_resolution(res_str):
    """Parse resolution string like '640x480' into tuple."""
    w, h = res_str.lower().split('x')
    return (int(w), int(h))


def init_camera(resolution, fps, quality, autofocus):
    global camera, jpeg_quality
    jpeg_quality = quality

    camera = Picamera2()

    config = camera.create_video_configuration(
        main={"size": resolution, "format": "RGB888"},
        controls={"FrameRate": fps}
    )
    camera.configure(config)

    # Enable autofocus on Camera 3
    if autofocus:
        try:
            camera.set_controls({"AfMode": 2})  # Continuous autofocus
            print("Autofocus: continuous")
        except Exception as e:
            print(f"Autofocus: not available ({e})")

    camera.start()
    # Let camera settle
    time.sleep(0.5)


def generate_frames():
    """Stream frames as MJPEG."""
    global frame_count, fps_start_time

    frame_count = 0
    fps_start_time = time.time()

    while True:
        frame = camera.capture_array()
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 5.0:
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}")
            frame_count = 0
            fps_start_time = time.time()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return '<html><body style="margin:0"><img src="/stream" style="width:100%"/></body></html>'


@app.route('/stream')
def stream():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pi 5 MJPEG streaming server')
    parser.add_argument('--resolution', '-r', default='640x480',
                        help='Resolution WxH (default: 640x480)')
    parser.add_argument('--fps', '-f', type=int, default=30,
                        help='Target frame rate (default: 30)')
    parser.add_argument('--quality', '-q', type=int, default=85,
                        help='JPEG quality 1-100 (default: 85)')
    parser.add_argument('--port', '-p', type=int, default=8080,
                        help='Server port (default: 8080)')
    parser.add_argument('--no-autofocus', action='store_true',
                        help='Disable autofocus')
    args = parser.parse_args()

    resolution = parse_resolution(args.resolution)
    print(f"Pi 5 Camera Stream")
    print(f"=" * 40)
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    print(f"Target FPS: {args.fps}")
    print(f"JPEG quality: {args.quality}")
    print()

    init_camera(resolution, args.fps, args.quality, not args.no_autofocus)
    print(f"Streaming at http://0.0.0.0:{args.port}")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
