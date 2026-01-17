#!/usr/bin/env python3
"""
Minimal MJPEG streaming server - no CV processing.
Optimized for Pi Zero W: lower resolution and JPEG quality for faster encoding.
Offloads all processing to client.

Performance: ~10-13 fps on Pi Zero W
For better performance, use stream_h264.py (hardware encoding, 25-30 fps)
"""

from flask import Flask, Response
from picamera2 import Picamera2
import cv2

app = Flask(__name__)
camera = None


def init_camera():
    global camera
    camera = Picamera2()
    # Lower resolution = faster JPEG encoding on Pi Zero W
    config = camera.create_video_configuration(
        main={"size": (320, 240), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()


def generate_frames():
    """Stream raw frames as fast as possible."""
    while True:
        frame = camera.capture_array()
        # Lower quality = faster encoding
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
    print("Initializing camera...")
    init_camera()
    print("Streaming raw video at http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, threaded=True)
