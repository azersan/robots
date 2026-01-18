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
import cv2

# Defaults
DEFAULT_PI_HOST = "192.168.4.80"
DEFAULT_PORT = 8080
DEFAULT_STREAM_PATH = "/stream"


def parse_args(description="CV app"):
    """Parse command-line arguments for video source selection."""
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
        # Remote stream (H264)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Connecting to {source}")

    return cap


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
