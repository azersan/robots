#!/usr/bin/env python3
"""
Hardware-accelerated H264 streaming with multi-client support.
Single rpicam-vid process shared across all connected clients.

Uses Pi's GPU encoder - expect 25-30 fps on Pi Zero W.
"""

import subprocess
import signal
import sys
from flask import Flask, Response
from threading import Thread, Lock
import time

app = Flask(__name__)

# Stream settings
WIDTH = 640
HEIGHT = 480
FPS = 30
BITRATE = 2000000  # 2 Mbps

# Shared stream state
running = True
rpicam_process = None
stream_thread = None


class StreamBuffer:
    """Thread-safe buffer for sharing stream data across clients."""

    def __init__(self, maxsize=30):
        self.buffer = []
        self.maxsize = maxsize
        self.lock = Lock()

    def put(self, chunk):
        with self.lock:
            self.buffer.append(chunk)
            if len(self.buffer) > self.maxsize:
                self.buffer.pop(0)

    def get_all(self):
        with self.lock:
            data = b''.join(self.buffer)
            self.buffer = []
            return data


# Global buffer for sharing stream
stream_buffer = StreamBuffer()


def stream_reader():
    """Background thread that reads from rpicam-vid."""
    global rpicam_process, running

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

    while running:
        chunk = rpicam_process.stdout.read(4096)
        if not chunk:
            break
        stream_buffer.put(chunk)

    rpicam_process.terminate()
    rpicam_process.wait()


def generate_h264():
    """Yield H264 chunks to client."""
    while running:
        data = stream_buffer.get_all()
        if data:
            yield data
        else:
            time.sleep(0.01)


@app.route('/')
def index():
    return '''
    <html>
    <head><title>Pi H264 Stream</title></head>
    <body style="margin:0;background:#000;color:#fff;padding:20px">
        <h1>H264 Stream (Multi-Client)</h1>
        <p>Stream running at /stream</p>
        <p>Supports multiple simultaneous viewers</p>
        <p>View with:</p>
        <ul>
            <li>VLC: vlc http://&lt;pi-ip&gt;:8080/stream</li>
            <li>OpenCV: python3 local_cv_h264.py</li>
        </ul>
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
    global running, rpicam_process
    print("\nShutting down...")
    running = False
    if rpicam_process:
        rpicam_process.terminate()
        rpicam_process.wait()
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Start stream reader in background
    stream_thread = Thread(target=stream_reader, daemon=True)
    stream_thread.start()

    print(f"Starting H264 stream at http://0.0.0.0:8080 (multi-client)")
    print(f"Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps")
    print(f"Bitrate: {BITRATE//1000}kbps")
    print()

    app.run(host='0.0.0.0', port=8080, threaded=True)
