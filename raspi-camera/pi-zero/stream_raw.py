#!/usr/bin/env python3
"""
MJPEG streaming server for Raspberry Pi.

Auto-detects Pi model and adjusts settings:
  - Pi 5: 640x480 @ 30fps, quality 85, autofocus enabled
  - Pi Zero/older: 320x240 @ 15fps, quality 70

Usage:
    python3 stream_raw.py                    # Auto-detect settings
    python3 stream_raw.py --resolution 1280x720 --quality 90
    python3 stream_raw.py --fps 60           # High frame rate mode
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


def is_pi5():
    """Detect if running on Pi 5."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            return 'Pi 5' in model
    except:
        return False


def parse_resolution(res_str):
    """Parse resolution string like '640x480' into tuple."""
    try:
        w, h = res_str.lower().split('x')
        return (int(w), int(h))
    except:
        return None


def init_camera(resolution, fps, quality, autofocus):
    global camera, jpeg_quality
    jpeg_quality = quality

    camera = Picamera2()

    config = camera.create_video_configuration(
        main={"size": resolution, "format": "RGB888"},
        controls={"FrameRate": fps}
    )
    camera.configure(config)

    # Enable autofocus on Camera 3 (Pi 5)
    if autofocus:
        try:
            camera.set_controls({"AfMode": 2})  # 2 = Continuous autofocus
            print("Autofocus: enabled (continuous)")
        except:
            print("Autofocus: not available")

    camera.start()


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
    parser = argparse.ArgumentParser(description='MJPEG streaming server')
    parser.add_argument('--resolution', '-r', default=None,
                        help='Resolution WxH (default: 640x480 on Pi 5, 320x240 on Zero)')
    parser.add_argument('--fps', '-f', type=int, default=None,
                        help='Target frame rate (default: 30 on Pi 5, 15 on Zero)')
    parser.add_argument('--quality', '-q', type=int, default=None,
                        help='JPEG quality 1-100 (default: 85 on Pi 5, 70 on Zero)')
    parser.add_argument('--port', '-p', type=int, default=8080,
                        help='Server port (default: 8080)')
    parser.add_argument('--no-autofocus', action='store_true',
                        help='Disable autofocus')
    args = parser.parse_args()

    # Auto-detect settings based on Pi model
    pi5 = is_pi5()
    if pi5:
        print("Detected: Raspberry Pi 5")
        defaults = {'resolution': (640, 480), 'fps': 30, 'quality': 85}
    else:
        print("Detected: Raspberry Pi Zero/other")
        defaults = {'resolution': (320, 240), 'fps': 15, 'quality': 70}

    resolution = parse_resolution(args.resolution) if args.resolution else defaults['resolution']
    fps = args.fps if args.fps else defaults['fps']
    quality = args.quality if args.quality else defaults['quality']
    autofocus = pi5 and not args.no_autofocus

    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    print(f"Target FPS: {fps}")
    print(f"JPEG quality: {quality}")
    print()

    print("Initializing camera...")
    init_camera(resolution, fps, quality, autofocus)
    print(f"Streaming at http://0.0.0.0:{args.port}")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
