"""One process owns the webcam + BOLT+; everything else talks to it over HTTP.

Humans open http://localhost:8770/ for the live annotated view + buttons.
Claude (or any script) uses the same endpoints:

  GET  /frame.jpg[?raw=1]           latest frame (annotated unless raw=1)
  GET  /state                       JSON: bolt detection, connection, camera ptz
  GET  /stream.mjpg                 annotated MJPEG stream
  POST /ptz?tilt=-5&pan=3&zoom=..   relative camera move (tilt_abs= etc. absolute)
  POST /bolt/roll?speed=50&dur=0.6  roll at the current heading (negative = back)
  POST /bolt/turn?deg=30            change heading
  POST /bolt/heading?deg=90         set absolute heading
  POST /bolt/stop
  POST /bolt/blink?on=0|1           toggle the LED blink used for detection
  POST /bolt/connect                (re)connect to the BOLT+

Detection: the BOLT+ blinks its LEDs; the blob that turns brighter in green
exactly when the LEDs are on is the robot (see detect()).
"""
import argparse
import json
import queue
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

from bolt import make_bolt

LED = Color(0, 255, 0)
PHASE_S = 0.35        # each LED on / off phase
SETTLE_S = 0.15       # wait after a phase change before trusting frames
PAIR_MAX_GAP_S = 1.0  # on/off frames further apart than this aren't compared
PROC_W = 640          # detection runs at this width
PTZ_PROPS = {"pan": cv2.CAP_PROP_PAN, "tilt": cv2.CAP_PROP_TILT, "zoom": cv2.CAP_PROP_ZOOM}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Camera(threading.Thread):
    """Owns the capture: keeps only the newest frame, applies PTZ requests."""

    def __init__(self, index):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cond = threading.Condition()
        self.frame, self.t, self.n = None, 0.0, 0
        self.ptz_q = queue.Queue()
        self.ptz = {}

    def run(self):
        while True:
            while not self.ptz_q.empty():
                name, val, absolute = self.ptz_q.get()
                cur = self.cap.get(PTZ_PROPS[name])
                ok = self.cap.set(PTZ_PROPS[name], val if absolute else cur + val)
                log(f"ptz {name}: {cur} -> {self.cap.get(PTZ_PROPS[name])} (set ok={ok})")
            self.ptz = {k: self.cap.get(v) for k, v in PTZ_PROPS.items()}
            ok, f = self.cap.read()
            if ok:
                with self.cond:
                    self.frame, self.t, self.n = f, time.time(), self.n + 1
                    self.cond.notify_all()


class Bolt(threading.Thread):
    """Owns the BLE connection: blinks the LEDs and applies drive commands."""

    def __init__(self):
        super().__init__(daemon=True)
        self.cmds = queue.Queue()
        self.status = "idle"
        self.blink = True
        self.led_on = False
        self.phase_t = 0.0   # when the current LED phase was acknowledged
        self.heading = 0
        self.want_connect = threading.Event()
        self.want_connect.set()

    def send(self, *cmd):
        self.cmds.put(cmd)

    def _set_leds(self, api, on):
        c = LED if on else Color(0, 0, 0)
        api.set_matrix_fill(0, 0, 7, 7, c)
        api.set_main_led(c)
        self.led_on, self.phase_t = on, time.time()

    def run(self):
        while True:
            self.want_connect.wait()
            self.want_connect.clear()
            self.status = "connecting"
            log("bolt: connecting")
            try:
                with SpheroEduAPI(make_bolt()) as api:
                    self.status = "connected"
                    log("bolt: connected")
                    self._loop(api)
            except Exception as e:
                self.status = f"error: {e!r}"[:120]
                log("bolt:", self.status)
                traceback.print_exc()
                time.sleep(3)
                self.want_connect.set()  # keep trying; robot may be asleep

    def _loop(self, api):
        stop_at, next_toggle = None, 0.0
        while not self.want_connect.is_set():
            try:
                cmd = self.cmds.get(timeout=0.02)
            except queue.Empty:
                cmd = None
            if cmd:
                kind = cmd[0]
                if kind == "roll":
                    api.set_heading(self.heading)
                    api.set_speed(cmd[1])
                    stop_at = time.time() + cmd[2]
                elif kind == "heading":
                    self.heading = int(cmd[1]) % 360
                    api.set_heading(self.heading)
                elif kind == "stop":
                    api.set_speed(0)
                    stop_at = None
            now = time.time()
            if stop_at and now > stop_at:
                api.set_speed(0)
                stop_at = None
            if self.blink and now >= next_toggle:
                self._set_leds(api, not self.led_on)
                next_toggle = self.phase_t + PHASE_S
            elif not self.blink and not self.led_on:
                self._set_leds(api, True)
        api.set_speed(0)


def detect(on, off):
    """Return ((x, y, radius, score) | None, mask) in proc coords."""
    d = on.astype(np.int16) - off.astype(np.int16)
    db, dg, dr = d[..., 0], d[..., 1], d[..., 2]
    # Got brighter, and green rose at least as much as red/blue (LED cores
    # blow out to white, so allow equality; the halo carries the colour).
    mask = ((dg > 35) & (dg >= dr) & (dg >= db)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    best = None
    area_max = 0.03 * mask.size  # bigger than this = lighting change / person
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 12 or area > area_max:
            continue
        blob = labels == i
        score = float(np.sum(dg[blob]) - 0.5 * np.sum(np.maximum(dr, db)[blob]))
        if best is None or score > best[3]:
            r = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) / 2
            best = (cents[i][0], cents[i][1], r, score)
    return best, mask


class Eyes(threading.Thread):
    """Runs detection on each new frame and renders the annotated view."""

    def __init__(self, cam, bolt):
        super().__init__(daemon=True)
        self.cam, self.bolt = cam, bolt
        self.lock = threading.Lock()
        self.view_jpg = self.raw_jpg = None
        self.view_n = 0
        self.det, self.det_t, self.mask = None, 0.0, None
        self.trail = deque(maxlen=30)
        self.fps = 0.0

    def state(self):
        b = self.bolt
        det = self.det
        return {
            "t": time.time(), "fps": round(self.fps, 1), "camera_ptz": self.cam.ptz,
            "bolt_link": {"status": b.status, "led_on": b.led_on, "blink": b.blink,
                          "heading": b.heading},
            "bolt": None if not det else {
                "x": round(det[0]), "y": round(det[1]), "radius": round(det[2]),
                "score": round(det[3]), "age_s": round(time.time() - self.det_t, 2)},
            "frame_size": None if self.cam.frame is None else
            [self.cam.frame.shape[1], self.cam.frame.shape[0]],
        }

    def run(self):
        on_f = off_f = None
        last_n, fps_t, fps_n = -1, time.time(), 0
        while True:
            with self.cam.cond:
                self.cam.cond.wait_for(lambda: self.cam.n != last_n, timeout=1)
                frame, ft, last_n = self.cam.frame, self.cam.t, self.cam.n
            if frame is None:
                continue
            fps_n += 1
            if time.time() - fps_t > 1:
                self.fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()

            scale = frame.shape[1] / PROC_W
            small = cv2.resize(frame, (PROC_W, int(frame.shape[0] / scale)))
            b = self.bolt
            if b.blink and b.status == "connected" and ft - b.phase_t > SETTLE_S:
                if b.led_on:
                    on_f = (ft, small)
                else:
                    off_f = (ft, small)
                if on_f and off_f and abs(on_f[0] - off_f[0]) < PAIR_MAX_GAP_S:
                    found, self.mask = detect(on_f[1], off_f[1])
                    if found:
                        x, y, r, s = found
                        self.det, self.det_t = (x * scale, y * scale, r * scale, s), time.time()
                        self.trail.append((int(x * scale), int(y * scale)))
            view = self.render(frame)
            ok1, vj = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
            ok2, rj = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with self.lock:
                self.view_jpg, self.raw_jpg = vj.tobytes(), rj.tobytes()
                self.view_n += 1

    def render(self, frame):
        view = frame.copy()
        for a, c in zip(self.trail, list(self.trail)[1:]):
            cv2.line(view, a, c, (0, 200, 255), 2)
        age = time.time() - self.det_t
        if self.det and age < 3:
            x, y, r, s = self.det
            x, y, r = int(x), int(y), int(max(r, 14)) + 8
            col = (0, 255, 0) if age < 1.0 else (0, 165, 255)
            cv2.circle(view, (x, y), r, col, 3)
            cv2.drawMarker(view, (x, y), col, cv2.MARKER_CROSS, 12, 2)
            cv2.putText(view, f"BOLT+? score {s:.0f}  {age:.1f}s ago",
                        (max(x - r, 0), max(y - r - 10, 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if self.mask is not None:
            w = view.shape[1]
            th = cv2.resize(cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), (256, 144))
            view[10:154, w - 266:w - 10] = th
            cv2.rectangle(view, (w - 266, 10), (w - 10, 154), (255, 255, 255), 1)
        b = self.bolt
        hud = (f"{self.fps:4.1f} fps | bolt {b.status} | LED {'on ' if b.led_on else 'off'}"
               f" | heading {b.heading} | tilt {self.cam.ptz.get('tilt')}")
        cv2.rectangle(view, (0, 0), (len(hud) * 11 + 20, 34), (0, 0, 0), -1)
        cv2.putText(view, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return view


PAGE = """<!doctype html><title>BOLT+ eyes</title>
<style>body{background:#111;color:#ddd;font:14px system-ui;margin:12px}
img{max-width:100%;border:1px solid #444}button{font:14px system-ui;padding:6px 12px;margin:2px}
#s{font:12px monospace;white-space:pre}</style>
<img src="/stream.mjpg"><div>
<b>Drive</b> <button onclick="p('/bolt/turn?deg=-30')">&#8630; -30</button>
<button onclick="p('/bolt/roll?speed=50&dur=0.6')">&#8593; fwd</button>
<button onclick="p('/bolt/roll?speed=-50&dur=0.6')">&#8595; back</button>
<button onclick="p('/bolt/turn?deg=30')">&#8631; +30</button>
<button onclick="p('/bolt/stop')">stop</button>
<button onclick="p('/bolt/blink?on=1')">blink on</button><button onclick="p('/bolt/blink?on=0')">blink off</button>
<button onclick="p('/bolt/connect')">reconnect</button>
&nbsp; <b>Camera</b> <button onclick="p('/ptz?tilt=5')">tilt up</button>
<button onclick="p('/ptz?tilt=-5')">tilt down</button>
<button onclick="p('/ptz?pan=-5')">pan left</button><button onclick="p('/ptz?pan=5')">pan right</button>
</div><div id="s"></div>
<script>function p(u){fetch(u,{method:'POST'})}
setInterval(async()=>{document.getElementById('s').textContent=
JSON.stringify(await (await fetch('/state')).json(),null,1)},1000)</script>"""


def make_handler(cam, bolt, eyes):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(200, PAGE, "text/html")
            if u.path == "/state":
                return self._send(200, eyes.state())
            if u.path == "/frame.jpg":
                with eyes.lock:
                    jpg = eyes.raw_jpg if q.get("raw") else eyes.view_jpg
                return self._send(200, jpg, "image/jpeg") if jpg else self._send(503, {"error": "no frame"})
            if u.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with eyes.lock:
                            n, jpg = eyes.view_n, eyes.view_jpg
                        if jpg is None or n == last:
                            time.sleep(0.01)
                            continue
                        last = n
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/ptz":
                    for name in PTZ_PROPS:
                        if name in q:
                            cam.ptz_q.put((name, float(q[name]), False))
                        if f"{name}_abs" in q:
                            cam.ptz_q.put((name, float(q[f"{name}_abs"]), True))
                elif u.path == "/bolt/roll":
                    bolt.send("roll", int(q.get("speed", 50)), float(q.get("dur", 0.6)))
                elif u.path == "/bolt/turn":
                    bolt.send("heading", bolt.heading + int(q.get("deg", 30)))
                elif u.path == "/bolt/heading":
                    bolt.send("heading", int(q["deg"]))
                elif u.path == "/bolt/stop":
                    bolt.send("stop")
                elif u.path == "/bolt/blink":
                    bolt.blink = q.get("on", "1") == "1"
                elif u.path == "/bolt/connect":
                    bolt.want_connect.set()
                else:
                    return self._send(404, {"error": "not found"})
            except (KeyError, ValueError) as e:
                return self._send(400, {"error": repr(e)})
            self._send(200, {"ok": True})

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--camera", type=int, default=0)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--no-bolt", action="store_true", help="camera only")
    args = p.parse_args()

    cam = Camera(args.camera)
    cam.start()
    bolt = Bolt()
    if args.no_bolt:
        bolt.want_connect.clear()
    bolt.start()
    eyes = Eyes(cam, bolt)
    eyes.start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(cam, bolt, eyes))
    log(f"eyes server on http://localhost:{args.port}/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
