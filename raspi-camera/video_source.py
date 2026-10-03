"""
Shared video source handling for CV apps.
Supports local webcam and remote Pi streams.

Usage:
    from video_source import parse_args, get_capture, reconnect

    args = parse_args()
    cap = get_capture(args)
    # ... in reconnect logic:
    cap = reconnect(args, cap)
"""

import argparse
import threading
import time
import cv2

# Defaults
DEFAULT_PI_HOST = "pibot5-2g.local"
DEFAULT_PORT = 8080
DEFAULT_STREAM_PATH = "/stream"


class FrameGrabber:
    """Wraps a cv2.VideoCapture in a background thread that always holds only
    the most recent frame.

    Network MJPEG streams buffer frames in the OS/ffmpeg pipeline. When the
    consumer (e.g. YOLO) is slower than the stream, plain cap.read() returns
    the OLDEST buffered frame, so latency grows without bound. This reader
    drains the capture continuously and keeps just the newest frame, so
    read() always returns "now" and stale frames are dropped instead of
    processed. Exposes the subset of the VideoCapture API the apps use.

    read() also waits briefly for a frame it hasn't returned before, so a
    consumer that's FASTER than the stream doesn't re-process duplicates.
    """

    def __init__(self, cap, warmup=3.0):
        self.cap = cap
        self.cond = threading.Condition()
        self.frame = None
        self.ret = False
        self.seq = 0            # increments on every new frame
        self._last_seq = 0      # last seq handed out by read()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        # Block briefly so callers see a ready capture (or a clean failure).
        start = time.time()
        while self.frame is None and self.running and (time.time() - start) < warmup:
            time.sleep(0.01)

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                with self.cond:
                    self.ret = False
                    self.cond.notify_all()
                time.sleep(0.005)  # avoid busy-spin on read failure
                continue
            # cap.read() returns a fresh array each call, so swapping the
            # reference under the lock needs no copy.
            with self.cond:
                self.ret = True
                self.frame = frame
                self.seq += 1
                self.cond.notify_all()

    def read(self, timeout=0.25):
        """Return (ret, frame), waiting up to `timeout` for an unseen frame.

        Returns immediately on stream failure (ret False) so reconnect logic
        in the apps still triggers; after the timeout it returns the latest
        frame even if already seen, so the caller's UI loop keeps running.
        """
        with self.cond:
            if self.ret and self.seq == self._last_seq:
                self.cond.wait(timeout)
            self._last_seq = self.seq
            return self.ret, self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def create_parser(description="CV app"):
    """Create an argument parser with video source options.

    Apps can add their own arguments before calling parse_args().
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--source', '-s',
        default='pi',
        help='Video source: "local" for webcam, "pi" for default Pi, or IP address (default: pi)'
    )
    parser.add_argument(
        '--local', '-l',
        action='store_true',
        help='Shorthand for --source local (use Mac webcam)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=DEFAULT_PORT,
        help=f'Stream port for remote sources (default: {DEFAULT_PORT})'
    )

    return parser


def parse_args(description="CV app", parser=None):
    """Parse command-line arguments for video source selection.

    Args:
        description: App description for help text
        parser: Optional pre-configured parser (from create_parser with extra args)
    """
    if parser is None:
        parser = create_parser(description)

    args = parser.parse_args()

    # --local flag overrides --source
    if args.local:
        args.source = 'local'

    return args


def get_source_url(args):
    """Get the video source URL or device ID."""
    if args.source == 'local':
        return 0  # Local webcam device ID

    # Remote stream
    if args.source == 'pi':
        host = DEFAULT_PI_HOST
    else:
        host = args.source

    return f"http://{host}:{args.port}{DEFAULT_STREAM_PATH}"


def get_capture(args):
    """Create and configure a VideoCapture for the given source."""
    source = get_source_url(args)

    if source == 0:
        # Local webcam
        cap = cv2.VideoCapture(0)
        print(f"Using local webcam")
    else:
        # Remote stream (MJPEG over HTTP)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Connecting to {source}")

    # Wrap in a latest-frame reader so slow consumers don't accumulate lag.
    return FrameGrabber(cap)


def reconnect(args, cap):
    """Release existing capture and create a new one."""
    if cap is not None:
        cap.release()
    return get_capture(args)


def is_local(args):
    """Check if using local webcam."""
    return args.source == 'local'


def get_source_description(args):
    """Get a human-readable description of the source."""
    if args.source == 'local':
        return "Local Webcam"
    elif args.source == 'pi':
        return f"Pi Stream ({DEFAULT_PI_HOST}:{args.port})"
    else:
        return f"Remote Stream ({args.source}:{args.port})"
