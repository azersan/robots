#!/usr/bin/env python3
"""
Simple MJPEG streaming server for Raspberry Pi Camera.
View the stream at http://<pi-ip>:8080 in any browser.
"""

from flask import Flask, Response
from picamera2 import Picamera2
import io

app = Flask(__name__)
camera = None


def init_camera():
    global camera
    camera = Picamera2()
    # Lower resolution for Pi Zero W performance
    config = camera.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()


def generate_frames():
    """Generator that yields MJPEG frames."""
    while True:
        stream = io.BytesIO()
        camera.capture_file(stream, format='jpeg')
        stream.seek(0)
        frame = stream.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Pi Camera Stream</title>
        <style>
            body {
                background: #1a1a1a;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <img src="/stream" />
    </body>
    </html>
    '''


@app.route('/stream')
def stream():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    print("Initializing camera...")
    init_camera()
    print("Starting stream at http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, threaded=True)
