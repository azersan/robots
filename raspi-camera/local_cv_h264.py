#!/usr/bin/env python3
"""
Local CV processing for H264 streams - runs on your Mac.
Pulls H264 video from Pi and does all processing locally.

Usage:
    python3 local_cv_h264.py              # Use default Pi address
    python3 local_cv_h264.py 192.168.4.80 # Specify Pi address
"""

import cv2
import numpy as np
import time
import sys

# Pi stream URL
PI_HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.80"
STREAM_URL = f"http://{PI_HOST}:8080/stream"

# Color tracking settings
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])
MIN_AREA = 1500  # Larger for 640x480


def process_frame(frame):
    """Detect red blobs and return annotated frame with tracking info."""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for red
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tracking info to return
    tracking = {"detected": False, "cx": 0, "cy": 0, "radius": 0, "area": 0}

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > MIN_AREA:
            tracking["detected"] = True
            tracking["area"] = area

            # Get bounding circle
            ((x, y), radius) = cv2.minEnclosingCircle(largest)
            tracking["radius"] = radius

            # Get centroid
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                tracking["cx"] = cx
                tracking["cy"] = cy

                # Draw circle and center
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Draw center line
                frame_center = frame.shape[1] // 2
                cv2.line(frame, (frame_center, 0), (frame_center, frame.shape[0]),
                        (100, 100, 100), 1)

    return frame, mask, tracking


def create_side_panel(mask, tracking, fps, frame_height):
    """Create an info panel showing mask and stats."""
    # Create panel
    panel_width = 250
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark gray background

    # Add mask preview at top (scaled to fit panel width)
    mask_w = panel_width - 20  # 10px margin on each side
    mask_h = int(mask_w * mask.shape[0] / mask.shape[1])
    mask_scaled = cv2.resize(mask, (mask_w, mask_h))
    mask_color = cv2.cvtColor(mask_scaled, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(mask_color, (0, 0), (mask_w-1, mask_h-1), (0, 255, 0), 1)

    # Place mask preview
    panel[10:10+mask_h, 10:10+mask_w] = mask_color

    # Add label
    cv2.putText(panel, "MASK VIEW", (10, 10 + mask_h + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Add stats
    y_offset = 10 + mask_h + 55

    # FPS
    cv2.putText(panel, f"FPS: {fps}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 30

    # Status
    if tracking["detected"]:
        status = "TRACKING"
        status_color = (0, 255, 0)
    else:
        status = "SEARCHING"
        status_color = (0, 100, 255)

    cv2.putText(panel, f"Status: {status}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
    y_offset += 30

    if tracking["detected"]:
        # Position
        cv2.putText(panel, f"Pos: ({tracking['cx']}, {tracking['cy']})", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30

        # Direction
        frame_center = 320  # Half of 640
        if tracking["cx"] < frame_center - 50:
            direction = "<< LEFT"
            dir_color = (0, 255, 255)
        elif tracking["cx"] > frame_center + 50:
            direction = "RIGHT >>"
            dir_color = (0, 255, 255)
        else:
            direction = "CENTER"
            dir_color = (0, 255, 0)

        cv2.putText(panel, f"Dir: {direction}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dir_color, 1)
        y_offset += 30

        # Size (rough distance indicator)
        cv2.putText(panel, f"Radius: {int(tracking['radius'])}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return panel


def main():
    print(f"Connecting to Pi H264 stream at {STREAM_URL}")
    print("Press 'q' to quit, 's' to save screenshot")
    print()

    # Open video stream with H264 decode hints
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: Could not connect to {STREAM_URL}")
        print("Make sure the Pi is running stream_h264.py")
        print()
        print("Troubleshooting:")
        print("  1. Check Pi is reachable: ping " + PI_HOST)
        print("  2. Test with VLC: vlc " + STREAM_URL)
        print("  3. Check stream is running on Pi")
        return

    print("Connected! Stream should appear shortly...")

    # FPS tracking
    fps_time = time.time()
    fps_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection, reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        # Process frame
        frame, mask, tracking = process_frame(frame)

        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()

        # Create side panel
        panel = create_side_panel(mask, tracking, fps, frame.shape[0])

        # Concatenate horizontally
        display = np.hstack([frame, panel])

        cv2.imshow("Pi Camera - H264 Local CV", display)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, display)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
