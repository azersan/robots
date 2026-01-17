#!/usr/bin/env python3
"""
Hardware-accelerated H264 streaming using Pi's GPU encoder.
Much faster than MJPEG - expect 25-30 fps on Pi Zero W.

Uses rpicam-vid for encoding, streams via HTTP for easy consumption.
Client connects to http://<pi-ip>:8080/stream

Note: Requires rpicam-apps to be installed (comes with recent Raspberry Pi OS)
"""

import subprocess
import signal
import sys
from flask import Flask, Response
from threading import Thread
import io

app = Flask(__name__)

# Stream settings
WIDTH = 640
HEIGHT = 480
FPS = 30
BITRATE = 2000000  # 2 Mbps

# Global process handle
rpicam_process = None


def start_rpicam():
    """Start rpicam-vid with H264 encoding to stdout."""
    global rpicam_process

    cmd = [
        'rpicam-vid',
        '-t', '0',                    # Run indefinitely
        '--width', str(WIDTH),
        '--height', str(HEIGHT),
        '--framerate', str(FPS),
        '--bitrate', str(BITRATE),
        '--profile', 'baseline',      # Better compatibility
        '--level', '4.2',
        '--inline',                   # Include headers in stream
        '-o', '-'                     # Output to stdout
    ]

    rpicam_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0
    )
    return rpicam_process


def generate_h264():
    """Yield H264 chunks from rpicam-vid."""
    proc = start_rpicam()

    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk
    finally:
        proc.terminate()
        proc.wait()


@app.route('/')
def index():
    return '''
    <html>
    <head><title>Pi H264 Stream</title></head>
    <body style="margin:0;background:#000">
        <video autoplay muted playsinline style="width:100%;height:100vh;object-fit:contain">
            <source src="/stream" type="video/h264">
        </video>
        <script>
            // Fallback message
            document.querySelector('video').onerror = function() {
                document.body.innerHTML = '<p style="color:white;padding:20px">H264 stream running at /stream<br>Use VLC or OpenCV to view</p>';
            };
        </script>
    </body>
    </html>
    '''


@app.route('/stream')
def stream():
    return Response(
        generate_h264(),
        mimetype='video/h264'
    )


def cleanup(sig, frame):
    global rpicam_process
    print("\nShutting down...")
    if rpicam_process:
        rpicam_process.terminate()
        rpicam_process.wait()
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"Starting H264 stream at http://0.0.0.0:8080")
    print(f"Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps")
    print(f"Bitrate: {BITRATE//1000}kbps")
    print()
    print("View options:")
    print("  - VLC: vlc http://<pi-ip>:8080/stream")
    print("  - OpenCV: See local_cv_h264.py")
    print()

    app.run(host='0.0.0.0', port=8080, threaded=True)
