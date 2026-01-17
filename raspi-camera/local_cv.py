#!/usr/bin/env python3
"""
Local CV processing - runs on your Mac.
Pulls video from Pi and does all processing locally.

Usage:
    python3 local_cv.py              # Use default Pi address
    python3 local_cv.py 192.168.4.80 # Specify Pi address
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
MIN_AREA = 500  # For 320x240 stream


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


def create_side_panel(mask, tracking, fps):
    """Create an info panel showing mask and stats."""
    h, w = mask.shape

    # Create panel (same height as frame, fixed width)
    panel_width = 200
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark gray background

    # Add mask preview at top (scaled to fit panel)
    mask_scaled = cv2.resize(mask, (180, 135))
    mask_color = cv2.cvtColor(mask_scaled, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(mask_color, (0, 0), (179, 134), (0, 255, 0), 1)
    panel[10:145, 10:190] = mask_color

    # Add label
    cv2.putText(panel, "MASK VIEW", (10, 165),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Add stats
    y_offset = 195

    # FPS
    cv2.putText(panel, f"FPS: {fps}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 25

    # Status
    if tracking["detected"]:
        status = "TRACKING"
        status_color = (0, 255, 0)
    else:
        status = "SEARCHING"
        status_color = (0, 100, 255)

    cv2.putText(panel, f"Status: {status}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    y_offset += 25

    if tracking["detected"]:
        # Position
        cv2.putText(panel, f"Pos: ({tracking['cx']}, {tracking['cy']})", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

        # Direction
        frame_center = 160  # Half of 320
        if tracking["cx"] < frame_center - 30:
            direction = "<< LEFT"
            dir_color = (0, 255, 255)
        elif tracking["cx"] > frame_center + 30:
            direction = "RIGHT >>"
            dir_color = (0, 255, 255)
        else:
            direction = "CENTER"
            dir_color = (0, 255, 0)

        cv2.putText(panel, f"Dir: {direction}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, dir_color, 1)
        y_offset += 25

        # Size (rough distance indicator)
        cv2.putText(panel, f"Radius: {int(tracking['radius'])}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return panel


def main():
    print(f"Connecting to Pi stream at {STREAM_URL}")
    print("Press 'q' to quit, 's' to save screenshot")
    print()

    # Open video stream
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print(f"ERROR: Could not connect to {STREAM_URL}")
        print("Make sure the Pi is running stream_raw.py or stream_h264.py")
        return

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
            cap = cv2.VideoCapture(STREAM_URL)
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
        panel = create_side_panel(mask, tracking, fps)

        # Combine frame and panel side by side
        # First upscale frame to 640x480
        frame_large = cv2.resize(frame, (640, 480))
        # Scale panel to match height
        panel_large = cv2.resize(panel, (250, 480))

        # Concatenate horizontally
        display = np.hstack([frame_large, panel_large])

        cv2.imshow("Pi Camera - Local CV", display)

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
