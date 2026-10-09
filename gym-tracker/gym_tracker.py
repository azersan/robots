#!/usr/bin/env python3
"""
Gym tracker: watch the camera, notice when someone is lifting, count reps and
sets, and read the weight off the plates.

  idle   - checks a frame every --idle-interval seconds for a person
  active - runs pose on every frame, counts reps, groups them into sets;
           a finished set goes to Claude (weigh.py) to name the lift and
           read the plates, then into the log (logbook.py)
  back to idle once nobody has been in frame for --absent-timeout seconds,
  which also ends the session (summary, optional sheet append)

Usage:
    python gym_tracker.py                     # Pi stream at pibot5-2g.local:8080
    python gym_tracker.py -s 192.168.4.56     # Pi stream at an IP
    python gym_tracker.py -s local            # this machine's webcam
    python gym_tracker.py -s clip.mp4         # a recorded video (every frame, video time)
    python gym_tracker.py -s picamera         # on the Pi itself, straight from the camera
    python gym_tracker.py --sheet             # also append the session to the workout log
"""

import argparse
import collections
import concurrent.futures
import os
import platform
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import features
from display import DEFAULT_HOST as DISPLAY_HOST, Display
from logbook import Logbook
from reps import make_counters
from sets import SetTracker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "raspi-camera"))
import video_source  # noqa: E402

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


# ---------------------------------------------------------------------------
# Video sources: each read() returns (ok, frame_bgr, t_seconds)
# ---------------------------------------------------------------------------

class StreamSource:
    """Pi MJPEG stream or local webcam, via the shared latest-frame grabber."""
    is_file = False

    def __init__(self, args):
        self.args = args
        self.cap = video_source.get_capture(args)
        print(f"Source: {video_source.get_source_description(args)}")

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame, time.monotonic()

    def reconnect(self):
        time.sleep(1)
        self.cap = video_source.reconnect(self.args, self.cap)

    def release(self):
        self.cap.release()


class FileSource:
    """A recorded video: every frame, timed by the video's own clock."""
    is_file = True

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            sys.exit(f"Could not open {path}")
        print(f"Source: {path}")

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def release(self):
        self.cap.release()


class PiCameraSource:
    """Direct Picamera2 capture, for running on the Pi (robot_server must not be running)."""
    is_file = False

    def __init__(self, size=(1280, 720), fps=30):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": size, "format": "RGB888"}, controls={"FrameRate": fps}))
        try:
            self.cam.set_controls({"AfMode": 2})  # continuous autofocus
        except Exception:
            pass
        self.cam.start()
        time.sleep(0.5)
        print("Source: Pi camera (direct)")

    def read(self):
        # Picamera2's "RGB888" is BGR byte order, which is what OpenCV wants.
        return True, self.cam.capture_array(), time.monotonic()

    def reconnect(self):
        time.sleep(1)

    def release(self):
        self.cam.stop()


def open_source(args):
    if args.source == "picamera":
        return PiCameraSource()
    if os.path.isfile(args.source):
        return FileSource(args.source)
    return StreamSource(args)


# ---------------------------------------------------------------------------

def load_landmarker(model):
    path = os.path.join(HERE, "models", f"pose_landmarker_{model}.task")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading pose model ({model})...")
        urllib.request.urlretrieve(MODEL_URLS[model], path)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def jpeg(frame, max_width=1280):
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def draw(frame, landmarks, hud):
    h, w = frame.shape[:2]
    if landmarks:
        pts = {i: (int(l.x * w), int(l.y * h)) for i, l in enumerate(landmarks) if l.visibility > 0.5}
        for a, b in SKELETON:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts.values():
            cv2.circle(frame, p, 3, (0, 200, 255), -1)
    for i, line in enumerate(hud):
        y = 30 + i * 28
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


class Tracker:
    def __init__(self, args):
        self.args = args
        self.log = Logbook(args.log_dir)
        self.weigher = None
        if not args.no_weigh:
            import weigh
            self.weigh = weigh
            self.weigher = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.display = None if args.no_display else Display(args.display_host)
        self.pending = []
        self.state = "idle"
        self._new_session()

    def _new_session(self):
        # About one frame a second from the last minute, so a finished set can
        # be shown with the bar at rest before and after it: frames taken
        # mid-rep are often blurred or have the plates out of the shot.
        self.recent = collections.deque(maxlen=60)
        self.counters = make_counters()
        self.sets = SetTracker(rest_gap=self.args.rest_gap, min_reps=self.args.min_reps)
        self.last_person = None
        self.sets_done = 0
        self.last_result = ""

    # -- finished sets -------------------------------------------------------

    def _context_frames(self, lift_set):
        """Frames around the set, when the bar is most likely sitting still.

        Two before and two after, spread out: any single moment may catch the
        lifter mid-load or still holding the bar (the 2026-10-09 session missed
        a change plate that way).
        """
        def nearest(target, lo, hi):
            frames = [(abs(t - target), t, jpg) for t, jpg in self.recent if lo <= t <= hi]
            return min(frames)[1:] if frames else None

        s, e = lift_set.t_start, lift_set.t_end
        picks = [("about 12 s before the first rep", nearest(s - 12, s - 20, s - 6)),
                 ("about 3 s before the first rep", nearest(s - 3, s - 6, s - 1.5)),
                 ("about 5 s after the last rep", nearest(e + 5, e + 3, e + 9)),
                 ("about 18 s after the last rep", nearest(e + 18, e + 12, e + 24))]
        return [(label, pick[1]) for label, pick in picks if pick]

    def finish(self, lift_set):
        if lift_set is None:
            return
        lift_set.context = self._context_frames(lift_set)
        self.sets_done += 1
        print(f"Set done: {lift_set.movement} x {lift_set.reps}"
              + ("" if self.weigher else " (not weighing)"))
        if self.display:
            self.display.set_done(lift_set.reps)
        if self.weigher is None:
            self.log.record(lift_set)
            return
        self.pending.append(self.weigher.submit(self._weigh_and_log, lift_set))

    def _weigh_and_log(self, lift_set):
        reading, error = None, None
        try:
            reading = self.weigh.read_set(lift_set)
        except Exception as e:  # CLI missing, not logged in, timed out: log the set anyway
            error = f"{type(e).__name__}: {e}"
        rec = self.log.record(lift_set, reading, error)
        if rec["rejected"]:
            if self.display:
                self.display.clear()
            print(f"  -> not a real set, discarded ({reading.notes})")
            return
        w = rec["weight_lb"]
        if self.display:
            self.display.set_read(rec["reps"], w, reading.confidence if reading else None)
        self.last_result = f"{rec['exercise']} {w:g} lb x {rec['reps']}" if w else f"{rec['exercise']} x {rec['reps']}"
        print(f"  -> {self.last_result}" + (f"  [{error}]" if error else
              f"  ({reading.confidence}: {reading.notes})"))

    def end_session(self):
        self.finish(self.sets.flush())
        concurrent.futures.wait(self.pending)
        self.pending = []
        self.log.push_session(self.args.sheet)
        self._new_session()

    # -- per frame -----------------------------------------------------------

    def step(self, frame, f):
        """Advance on one analysed frame. Returns HUD lines."""
        if f.person:
            self.last_person = f.t
        if self.state == "active" and (not self.recent or f.t - self.recent[-1][0] >= 1.0):
            self.recent.append((f.t, jpeg(frame)))

        if self.state == "idle":
            if f.person:
                self.state = "active"
                print(f"[{time.strftime('%H:%M:%S')}] Person in frame - tracking")
            return ["IDLE - watching for a person"]

        for counter in self.counters:
            event = counter.update(f)
            if event is None:
                continue
            self.finish(self.sets.on_rep(event, jpeg(frame)))
            cur = self.sets.current
            if cur is not None and cur.movement == event.movement:
                # Live count is cycles; a hinge set's extra put-down cycle is
                # only dropped when the set is closed.
                print(f"  {event.movement} rep {len(cur.events)} ({event.t_end - event.t_start:.1f}s)")
                # The clock starts counting at the second rep: a lone first
                # "rep" is often just crouching or reaching, and gets dropped.
                if self.display and len(cur.events) >= self.args.min_reps:
                    self.display.rep(event.movement, len(cur.events))
        self.finish(self.sets.tick(f.t))

        if self.last_person is None or f.t - self.last_person > self.args.absent_timeout:
            print(f"[{time.strftime('%H:%M:%S')}] Nobody in frame - session over")
            self.state = "idle"
            self.end_session()
            return ["IDLE - watching for a person"]

        cur = self.sets.current
        hud = [f"ACTIVE  sets this session: {self.sets_done}"]
        hud.append(f"{cur.movement}: {len(cur.events)} reps" if cur else "waiting for a rep")
        if self.last_result:
            hud.append(f"last: {self.last_result}")
        return hud


def main():
    parser = video_source.create_parser("Gym tracker: reps, sets, and weight from the camera")
    parser.add_argument("--model", choices=["lite", "full"],
                        default="lite" if platform.machine() in ("aarch64", "armv7l") else "full",
                        help="Pose model (default: lite on the Pi, full elsewhere)")
    parser.add_argument("--idle-interval", type=float, default=2.0,
                        help="Seconds between person checks while idle (default 2)")
    parser.add_argument("--absent-timeout", type=float, default=60.0,
                        help="Seconds with nobody in frame before the session ends (default 60)")
    parser.add_argument("--rest-gap", type=float, default=25.0,
                        help="Seconds without a rep that ends a set (default 25)")
    parser.add_argument("--min-reps", type=int, default=2,
                        help="Shorter sets are dropped as noise (default 2)")
    parser.add_argument("--no-weigh", action="store_true", help="Skip the Claude plate reading")
    parser.add_argument("--sheet", action="store_true",
                        help="Append each session to the workout-log Google Sheet")
    parser.add_argument("--headless", action="store_true", help="No preview window")
    parser.add_argument("--display-host", default=DISPLAY_HOST,
                        help="Garage LED clock (AWTRIX) for live reps/results")
    parser.add_argument("--no-display", action="store_true", help="Don't post to the LED clock")
    parser.add_argument("--log-dir", default=os.path.join(HERE, "logs"))
    args = video_source.parse_args(parser=parser)

    source = open_source(args)
    landmarker = load_landmarker(args.model)
    tracker = Tracker(args)
    print(f"Pose model: {args.model}. Press q in the window (or Ctrl-C) to quit.")

    last_check = float("-inf")
    last_ts = 0
    hud = ["IDLE - watching for a person"]
    try:
        while True:
            ok, frame, t = source.read()
            if not ok or frame is None:
                if source.is_file:
                    break
                print("Lost the video source, reconnecting...")
                source.reconnect()
                continue

            landmarks = None
            if tracker.state == "active" or t - last_check >= args.idle_interval:
                last_check = t
                ts = max(int(t * 1000), last_ts + 1)
                last_ts = ts
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                h, w = frame.shape[:2]
                hud = tracker.step(frame, features.compute(landmarks, world, w, h, t))

            if not args.headless:
                draw(frame, landmarks, hud)
                cv2.imshow("gym tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        tracker.end_session()
        source.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
