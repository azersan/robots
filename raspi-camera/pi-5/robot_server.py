#!/usr/bin/env python3
"""
Combined camera + motor control server for Raspberry Pi 5.

Serves the MJPEG video stream AND accepts drive commands over HTTP, so only
one process needs to run on the Pi. Replaces stream.py (same port/path, so
existing Mac stream clients keep working unchanged).

Endpoints:
  GET  /         - HTML viewer
  GET  /stream   - MJPEG video stream
  POST /drive    - {"steer": -1..1, "speed": -1..1}   curve while driving
  POST /tank     - {"left": -1..1, "right": -1..1}    direct per-side control
  POST /stop     - stop motors
  GET  /health   - {"ok": true}

Safety:
  A watchdog stops the motors if no drive/tank command arrives within
  COMMAND_TIMEOUT seconds, so a WiFi drop or client crash halts the robot
  instead of letting it run away on its last command.

Usage:
    python3 robot_server.py                  # 640x480 @ 30fps
    python3 robot_server.py -r 1280x720 -f 15
    python3 robot_server.py --no-motors      # stream only (e.g. bench testing)
"""

from flask import Flask, Response, request, jsonify
from picamera2 import Picamera2
import cv2
import argparse
import math
import signal
import threading
import time

from motors import MotorController, NEUTRAL

app = Flask(__name__)

# --- Camera state ---
camera = None
broadcaster = None

# --- Motor state ---
motors = None
motor_lock = threading.Lock()        # serialize motor access (handlers + watchdog)
last_command_time = 0.0              # monotonic time of last drive/tank command
COMMAND_TIMEOUT = 0.5                # auto-stop if no command within this window (s)
SPEED_MAX = 130                      # max us offset from neutral mapped from speed=1.0
DEADBAND_US = 80                     # ESC ignores offsets smaller than ~75us


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def command_to_offset(v):
    """Map a normalized command (-1..1) to a us offset from neutral.

    The tinyESC has a ~75us deadband around neutral where nothing moves, so
    a linear map wastes the bottom half of the command range. Instead, any
    nonzero command starts just past the deadband and scales up to SPEED_MAX.
    """
    if abs(v) < 0.02:
        return 0.0
    return math.copysign(DEADBAND_US + abs(v) * (SPEED_MAX - DEADBAND_US), v)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class FrameBroadcaster:
    """Single capture+encode loop shared by all /stream clients.

    Picamera2 isn't built for concurrent capture calls from multiple Flask
    threads, and per-client JPEG encoding multiplies CPU load (which worsens
    the software PWM jitter). One thread captures and encodes the latest
    frame; every client generator just waits for the next one.
    """

    def __init__(self, camera, quality):
        self.camera = camera
        self.quality = quality
        self.cond = threading.Condition()
        self.jpeg = None
        self.seq = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            frame = self.camera.capture_array()
            ok, buffer = cv2.imencode('.jpg', frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                continue
            with self.cond:
                self.jpeg = buffer.tobytes()
                self.seq += 1
                self.cond.notify_all()

    def frames(self):
        """Yield each new JPEG once (per-client generator)."""
        last_seq = 0
        while True:
            with self.cond:
                while self.seq == last_seq:
                    self.cond.wait(1.0)
                last_seq = self.seq
                data = self.jpeg
            yield data


def init_camera(resolution, fps, quality, autofocus):
    global camera, broadcaster

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": resolution, "format": "RGB888"},
        controls={"FrameRate": fps},
    )
    camera.configure(config)

    if autofocus:
        try:
            camera.set_controls({"AfMode": 2})  # continuous
            print("Autofocus: continuous")
        except Exception as e:
            print(f"Autofocus: not available ({e})")

    camera.start()
    time.sleep(0.5)  # let camera settle
    broadcaster = FrameBroadcaster(camera, quality)


def generate_frames():
    for jpeg in broadcaster.frames():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')


@app.route('/')
def index():
    return '<html><body style="margin:0"><img src="/stream" style="width:100%"/></body></html>'


@app.route('/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------
# Motors
# ---------------------------------------------------------------------------

def _mark_command():
    global last_command_time
    last_command_time = time.monotonic()


def motor_watchdog():
    """Stop the motors if commands stop arriving (deadman switch).

    Starts with stopped=False so the neutral pulses from motor init get
    disarmed after the first idle period (otherwise the motors twitch at
    rest from boot until the first drive command comes and goes).
    """
    stopped = False
    while True:
        time.sleep(0.1)
        if motors is None:
            continue
        # Check idleness and disarm under the lock: handlers mark the
        # timestamp inside the same lock BEFORE touching the motors, so we
        # can't disarm a command that just landed.
        with motor_lock:
            idle = (time.monotonic() - last_command_time) > COMMAND_TIMEOUT
            if idle and not stopped:
                motors.disarm()  # cease pulses so motors don't twitch at rest
                stopped = True
            elif not idle:
                stopped = False


@app.route('/drive', methods=['POST'])
def drive():
    if motors is None:
        return jsonify(ok=False, error="motors disabled"), 503
    data = request.get_json(force=True, silent=True) or {}
    steer = clamp(float(data.get('steer', 0.0)), -1.0, 1.0)
    speed = clamp(float(data.get('speed', 0.0)), -1.0, 1.0)
    with motor_lock:
        _mark_command()
        motors.drive(steer=steer, speed=command_to_offset(speed))
    return jsonify(ok=True, steer=steer, speed=speed)


@app.route('/tank', methods=['POST'])
def tank():
    if motors is None:
        return jsonify(ok=False, error="motors disabled"), 503
    data = request.get_json(force=True, silent=True) or {}
    left = clamp(float(data.get('left', 0.0)), -1.0, 1.0)
    right = clamp(float(data.get('right', 0.0)), -1.0, 1.0)
    with motor_lock:
        _mark_command()
        motors.set_motors(NEUTRAL + command_to_offset(left),
                          NEUTRAL + command_to_offset(right))
    return jsonify(ok=True, left=left, right=right)


@app.route('/stop', methods=['POST'])
def stop():
    if motors is None:
        return jsonify(ok=False, error="motors disabled"), 503
    with motor_lock:
        _mark_command()
        motors.stop()
    return jsonify(ok=True)


@app.route('/health')
def health():
    return jsonify(ok=True, motors=motors is not None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_resolution(res_str):
    w, h = res_str.lower().split('x')
    return (int(w), int(h))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pi 5 combined camera + control server')
    parser.add_argument('--resolution', '-r', default='640x480', help='WxH (default 640x480)')
    parser.add_argument('--fps', '-f', type=int, default=30, help='Frame rate (default 30)')
    parser.add_argument('--quality', '-q', type=int, default=85, help='JPEG quality (default 85)')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Server port (default 8080)')
    parser.add_argument('--no-autofocus', action='store_true', help='Disable autofocus')
    parser.add_argument('--no-motors', action='store_true', help='Stream only, no motor control')
    args = parser.parse_args()

    resolution = parse_resolution(args.resolution)
    print("Pi 5 Robot Server")
    print("=" * 40)
    print(f"Resolution: {resolution[0]}x{resolution[1]} @ {args.fps}fps")

    if not args.no_motors:
        motors = MotorController()
        _mark_command()
        threading.Thread(target=motor_watchdog, daemon=True).start()
        print(f"Motors: enabled (watchdog stops after {COMMAND_TIMEOUT}s idle)")
    else:
        print("Motors: DISABLED (--no-motors)")

    init_camera(resolution, args.fps, args.quality, not args.no_autofocus)
    print(f"Streaming + control at http://0.0.0.0:{args.port}")

    # `kill` (SIGTERM) would normally skip the finally block below; convert
    # it to a normal exit so the motors get disarmed and GPIO released.
    def graceful_exit(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, graceful_exit)

    try:
        # threaded=True so the long-lived /stream generator doesn't block
        # control POSTs (and vice versa).
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    finally:
        if motors is not None:
            motors.cleanup()
